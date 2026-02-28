import { proxyGet } from "@/lib/server/rvc-proxy";

export const runtime = "nodejs";

export async function GET(request: Request) {
  return proxyGet("/api/jobs", request);
}
