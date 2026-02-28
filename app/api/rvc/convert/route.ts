import { proxyPostFormData } from "@/lib/server/rvc-proxy";

export const runtime = "nodejs";

export async function POST(request: Request) {
  return proxyPostFormData(request, "/api/convert");
}
