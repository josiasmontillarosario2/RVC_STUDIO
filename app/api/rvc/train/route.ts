import { proxyPostJson } from "@/lib/server/rvc-proxy";

export const runtime = "nodejs";

export async function POST(request: Request) {
  return proxyPostJson(request, "/api/train");
}
