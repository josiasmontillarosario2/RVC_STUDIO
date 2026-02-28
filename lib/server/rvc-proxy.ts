import "server-only";

const DEFAULT_RVC_API_BASE_URL = "http://127.0.0.1:8000";

function getRvcApiBaseUrl(): string {
  return (process.env.RVC_API_BASE_URL || DEFAULT_RVC_API_BASE_URL).replace(/\/+$/, "");
}

function buildUpstreamUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getRvcApiBaseUrl()}${normalizedPath}`;
}

function copyHeaders(source: Headers, extra?: HeadersInit): Headers {
  const headers = new Headers();
  for (const [key, value] of source.entries()) {
    const lower = key.toLowerCase();
    if (
      lower === "host" ||
      lower === "connection" ||
      lower === "content-length" ||
      lower === "transfer-encoding"
    ) {
      continue;
    }
    headers.set(key, value);
  }
  if (extra) {
    const add = new Headers(extra);
    for (const [key, value] of add.entries()) {
      headers.set(key, value);
    }
  }
  return headers;
}

function toClientResponse(upstream: Response): Response {
  const headers = new Headers();
  for (const [key, value] of upstream.headers.entries()) {
    const lower = key.toLowerCase();
    if (lower === "content-encoding" || lower === "transfer-encoding" || lower === "connection") {
      continue;
    }
    headers.set(key, value);
  }

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers,
  });
}

async function fetchUpstream(input: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown proxy error";
    return Response.json(
      {
        detail: `Failed to reach RVC backend at ${getRvcApiBaseUrl()}: ${message}`,
      },
      { status: 502 }
    );
  }
}

export async function proxyGet(path: string, request?: Request): Promise<Response> {
  const upstream = await fetchUpstream(buildUpstreamUrl(path), {
    method: "GET",
    headers: request ? copyHeaders(request.headers) : undefined,
    cache: "no-store",
  });
  return toClientResponse(upstream);
}

export async function proxyPostJson(request: Request, path: string): Promise<Response> {
  const bodyText = await request.text();
  const upstream = await fetchUpstream(buildUpstreamUrl(path), {
    method: "POST",
    headers: copyHeaders(request.headers, { "content-type": "application/json" }),
    body: bodyText,
    cache: "no-store",
  });
  return toClientResponse(upstream);
}

export async function proxyPostFormData(request: Request, path: string): Promise<Response> {
  const formData = await request.formData();
  const headers = copyHeaders(request.headers);
  headers.delete("content-type");

  const upstream = await fetchUpstream(buildUpstreamUrl(path), {
    method: "POST",
    headers,
    body: formData,
    cache: "no-store",
  });
  return toClientResponse(upstream);
}
