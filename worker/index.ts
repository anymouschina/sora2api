import { Container, getContainer } from "@cloudflare/containers";

export interface Env {
  DB: D1Database;
  SORA_API: DurableObjectNamespace<Container>;
  D1_PROXY_BASE_URL?: string;
  D1_PROXY_TOKEN?: string;
  SORA2API_API_KEY?: string;
}

export class SoraApi extends Container<Env> {
  defaultPort = 8000;

  constructor(ctx: any, env: Env) {
    super(ctx, env);

    const envVars: Record<string, string> = {};
    if (env.D1_PROXY_BASE_URL) envVars.D1_PROXY_BASE_URL = env.D1_PROXY_BASE_URL;
    if (env.D1_PROXY_TOKEN) envVars.D1_PROXY_TOKEN = env.D1_PROXY_TOKEN;
    if (env.SORA2API_API_KEY) envVars.SORA2API_API_KEY = env.SORA2API_API_KEY;

    this.envVars = envVars;
  }
}

function requireBearer(request: Request, secret: string): boolean {
  const auth = request.headers.get("authorization") || "";
  return auth === `Bearer ${secret}`;
}

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/__d1/query")) {
      if (env.D1_PROXY_TOKEN && !requireBearer(request, env.D1_PROXY_TOKEN)) {
        return new Response("Unauthorized", { status: 401 });
      }

      const body = await request.json<any>().catch(() => ({}));
      const sql = body.sql as string;
      const params = (body.params as any[]) || [];
      const method = (body.method as "all" | "run") || "all";

      if (!sql) {
        return Response.json({ success: false, error: "Missing sql" }, { status: 400 });
      }

      try {
        const stmt = env.DB.prepare(sql).bind(...params);
        const result = method === "run" ? await stmt.run() : await stmt.all();
        return Response.json({ success: true, result });
      } catch (err: any) {
        return Response.json(
          { success: false, error: err?.message || String(err) },
          { status: 500 }
        );
      }
    }

    const container = getContainer(env.SORA_API);
    return container.fetch(request);
  },
};
