type StatusNoticeProps = {
  kind?: "info" | "success" | "error" | "warning";
  children: React.ReactNode;
};

export default function StatusNotice({ kind = "info", children }: StatusNoticeProps) {
  return <div className={`notice notice--${kind}`} role={kind === "error" ? "alert" : "status"}>{children}</div>;
}
