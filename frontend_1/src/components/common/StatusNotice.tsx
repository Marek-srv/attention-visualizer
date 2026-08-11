import type { ReactNode } from "react";

export type NoticeKind = "info" | "success" | "warning" | "error";

type StatusNoticeProps = {
  children: ReactNode;
  kind?: NoticeKind;
  title?: string;
};

export default function StatusNotice({ children, kind = "info", title }: StatusNoticeProps) {
  return (
    <div
      className={`status-notice status-notice--${kind}`}
      role={kind === "error" ? "alert" : "status"}
      aria-live={kind === "error" ? "assertive" : "polite"}
    >
      {title ? <strong>{title}</strong> : null}
      <div>{children}</div>
    </div>
  );
}
