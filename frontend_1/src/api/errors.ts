import type { ValidationIssue } from "../types/api";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validationMessage(issue: ValidationIssue): string {
  const field = issue.loc
    .filter((part) => part !== "body")
    .map(String)
    .join(".");
  return field ? `${field}: ${issue.msg}` : issue.msg;
}

function isValidationIssue(value: unknown): value is ValidationIssue {
  if (!isRecord(value) || typeof value.msg !== "string" || !Array.isArray(value.loc)) {
    return false;
  }
  return value.loc.every((part) => typeof part === "string" || typeof part === "number");
}

export function readableApiError(payload: unknown, fallback: string): string {
  if (typeof payload === "string" && payload.trim()) {
    return payload.trim();
  }
  if (!isRecord(payload)) {
    return fallback;
  }

  const detail = payload.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (Array.isArray(detail)) {
    const messages = detail.filter(isValidationIssue).map(validationMessage);
    if (messages.length > 0) {
      return messages.join(" · ");
    }
  }
  if (isRecord(detail) && typeof detail.message === "string" && detail.message.trim()) {
    return detail.message.trim();
  }
  if (typeof payload.message === "string" && payload.message.trim()) {
    return payload.message.trim();
  }
  return fallback;
}

export class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : isRecord(error) && error.name === "AbortError";
}
