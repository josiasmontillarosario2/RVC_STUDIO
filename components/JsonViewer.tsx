interface JsonViewerProps {
  data: unknown;
  title?: string;
}

const JsonViewer = ({ data, title = "Response" }: JsonViewerProps) => {
  if (data === null || data === undefined) return null;

  return (
    <div className="mt-4">
      <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-2">{title}</h3>
      <pre className="bg-secondary rounded-md p-4 text-sm font-mono text-foreground overflow-auto max-h-64 border border-border">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
};

export default JsonViewer;
