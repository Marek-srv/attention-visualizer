import { ApiError, isAbortError, readableApiError } from "./errors";

export interface ApiRequestOptions {
  signal?: AbortSignal;
}

export interface ApiClientOptions {
  baseUrl?: string;
  fetchImplementation?: typeof fetch;
}

interface RequestConfiguration<TBody> extends ApiRequestOptions {
  method: "GET" | "POST";
  body?: TBody;
}

export function resolveApiBaseUrl(configuredValue = import.meta.env.VITE_API_BASE_URL): string {
  if (typeof configuredValue !== "string") {
    return "";
  }
  return configuredValue.trim().replace(/\/+$/, "");
}

function joinUrl(baseUrl: string, path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${normalizedPath}`;
}

async function parsePayload(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImplementation: typeof fetch;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = resolveApiBaseUrl(options.baseUrl);
    this.fetchImplementation = options.fetchImplementation ?? globalThis.fetch.bind(globalThis);
  }

  createController(): AbortController {
    return new AbortController();
  }

  get<TResponse>(path: string, options: ApiRequestOptions = {}): Promise<TResponse> {
    return this.request<TResponse, never>(path, { method: "GET", ...options });
  }

  post<TResponse, TBody>(path: string, body: TBody, options: ApiRequestOptions = {}): Promise<TResponse> {
    return this.request<TResponse, TBody>(path, { method: "POST", body, ...options });
  }

  private async request<TResponse, TBody>(
    path: string,
    configuration: RequestConfiguration<TBody>,
  ): Promise<TResponse> {
    const hasBody = configuration.body !== undefined;
    let response: Response;
    try {
      response = await this.fetchImplementation(joinUrl(this.baseUrl, path), {
        method: configuration.method,
        signal: configuration.signal,
        headers: hasBody ? { "Content-Type": "application/json" } : undefined,
        body: hasBody ? JSON.stringify(configuration.body) : undefined,
      });
    } catch (error) {
      if (isAbortError(error)) {
        throw error;
      }
      throw new ApiError("Backend disconnected. Check that FastAPI is running and try again.", 0, error);
    }

    const payload = await parsePayload(response);
    if (!response.ok) {
      const fallback = response.statusText
        ? `Request failed (${response.status} ${response.statusText}).`
        : `Request failed (${response.status}).`;
      throw new ApiError(readableApiError(payload, fallback), response.status, payload);
    }
    return payload as TResponse;
  }
}

export const apiClient = new ApiClient();
