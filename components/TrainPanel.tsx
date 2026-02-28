"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  startTrain,
  getJobStatus,
  downloadJobOutput,
  uploadDataset,
  type TrainRequest,
  type JobStatus,
  type UploadDatasetResponse,
} from "@/lib/api";
import JsonViewer from "@/components/JsonViewer";

const STORAGE_KEY = "rvc_train_form";

const defaultValues: TrainRequest = {
  dataset_dir: "",
  exp: "",
  sr: "40k",
  version: "v2",
  if_f0: 1,
  np: 2,
  f0_method: "rmvpe_gpu",
  gpus: "0",
  gpus_rmvpe: "0",
  batch_size: 4,
  save_every_epoch: 5,
  total_epoch: 10,
  early_stop_patience: 8,
  early_stop_min_delta: 0.05,
  early_stop_metric: "loss_mel",
  save_every_weights: true,
  copy_to_models: true,
  device: "cpu",
  is_half: false,
};

const TrainPanel = () => {
  const [form, setForm] = useState<TrainRequest>(defaultValues);
  const [hasLoadedForm, setHasLoadedForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [uploadDatasetError, setUploadDatasetError] = useState<string | null>(null);
  const [uploadDatasetResult, setUploadDatasetResult] = useState<UploadDatasetResponse | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const [response, setResponse] = useState<unknown>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const datasetFolderInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    try {
      const s = localStorage.getItem(STORAGE_KEY);
      if (s) {
        setForm({ ...defaultValues, ...JSON.parse(s) });
      }
    } catch {
      // Ignore invalid localStorage payloads and keep defaults
    } finally {
      setHasLoadedForm(true);
    }
  }, []);

  useEffect(() => {
    if (!hasLoadedForm) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
  }, [form, hasLoadedForm]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setPolling(false);
  }, []);

  const startPolling = useCallback((id: string) => {
    setPolling(true);
    pollingRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(id);
        setJobStatus(status);
        setResponse(status);
        if (status.status === "completed" || status.status === "failed") {
          stopPolling();
          if (status.status === "failed") {
            setError(status.error || "Training failed");
          }
        }
      } catch (err: unknown) {
        // Don't stop polling on network errors
        console.error("Poll error:", err);
      }
    }, 2000);
  }, [stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const handleSubmit = async () => {
    if (!form.exp.trim()) {
      setError("exp is required");
      return;
    }
    if (!form.dataset_dir.trim()) {
      setError("Escribe dataset_dir o sube archivos primero");
      return;
    }
    setError(null);
    setLoading(true);
    setResponse(null);
    setJobStatus(null);
    abortRef.current = new AbortController();

    try {
      const res = await startTrain(form, abortRef.current.signal);
      setResponse(res);
      setJobId(res.job_id);
      startPolling(res.job_id);
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleUploadDatasetFiles = async (files: File[]) => {
    if (!form.exp.trim()) {
      setUploadDatasetError("exp is required before uploading a dataset");
      return;
    }
    if (files.length === 0) {
      setUploadDatasetError("Select a folder with audio files");
      return;
    }

    setUploadingDataset(true);
    setUploadDatasetError(null);
    setUploadDatasetResult(null);

    try {
      const result = await uploadDataset(form.exp.trim(), files, true);
      setUploadDatasetResult(result);
      setForm((f) => ({ ...f, dataset_dir: result.dataset_dir }));
    } catch (err: unknown) {
      setUploadDatasetError(err instanceof Error ? err.message : "Dataset upload failed");
    } finally {
      setUploadingDataset(false);
    }
  };

  const handleDatasetFilesChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    await handleUploadDatasetFiles(files);
    e.target.value = "";
  };

  const openDatasetFolderPicker = () => {
    if (!form.exp.trim()) {
      setUploadDatasetError("exp is required before selecting a folder");
      return;
    }
    setUploadDatasetError(null);
    datasetFolderInputRef.current?.click();
  };

  const handleDownload = async () => {
    if (!jobId) return;
    try {
      const blob = await downloadJobOutput(jobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${form.exp}_output`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Download failed");
    }
  };

  const copyJobId = () => {
    if (jobId) navigator.clipboard.writeText(jobId);
  };

  const reset = () => {
    stopPolling();
    setForm({ ...defaultValues });
    setUploadingDataset(false);
    setUploadDatasetError(null);
    setUploadDatasetResult(null);
    setJobId(null);
    setJobStatus(null);
    setError(null);
    setResponse(null);
    setLoading(false);
    localStorage.removeItem(STORAGE_KEY);
  };

  const update = <K extends keyof TrainRequest>(key: K, val: TrainRequest[K]) => {
    setForm((f) => ({ ...f, [key]: val }));
  };

  const disabled = loading || polling;
  const uploadDisabled = disabled || uploadingDataset;

  const statusColor = (s?: string) => {
    switch (s) {
      case "completed": return "text-success";
      case "failed": return "text-destructive";
      case "running": return "text-warning";
      default: return "text-info";
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      {jobId && (
        <div className="bg-secondary rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground">JOB</span>
              <code className="text-sm font-mono text-foreground">{jobId}</code>
              <button onClick={copyJobId} className="text-xs text-primary hover:underline">Copy</button>
            </div>
            <div className="flex items-center gap-3">
              {jobStatus && (
                <span className={`text-sm font-mono font-semibold uppercase ${statusColor(jobStatus.status)}`}>
                  {polling && <span className="inline-block w-2 h-2 rounded-full bg-current mr-2 animate-pulse-dot" />}
                  {jobStatus.status}
                </span>
              )}
              {polling && (
                <button onClick={stopPolling} className="text-xs px-3 py-1 rounded bg-muted text-muted-foreground hover:text-foreground border border-border transition-colors">
                  Stop Polling
                </button>
              )}
              {jobStatus?.status === "completed" && (
                <span className="text-xs text-success font-mono">Training complete — check your models folder</span>
              )}
            </div>
          </div>
          {jobStatus?.started_at && (
            <div className="mt-2 text-xs text-muted-foreground font-mono">
              Started: {jobStatus.started_at}
              {jobStatus.finished_at && <> · Finished: {jobStatus.finished_at}</>}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {uploadDatasetError && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 text-sm text-destructive">
          Dataset upload: {uploadDatasetError}
        </div>
      )}

      {uploadDatasetResult && (
        <div className="bg-primary/10 border border-primary/30 rounded-lg p-3 text-sm text-primary">
          <div className="font-mono">Dataset uploaded to: {uploadDatasetResult.dataset_dir}</div>
          <div className="text-xs mt-1">Files: {uploadDatasetResult.count}</div>
        </div>
      )}

      {/* Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Field label="exp *" full>
          <input value={form.exp} onChange={(e) => update("exp", e.target.value)} disabled={disabled}
            className="form-input" placeholder="experiment_name" />
        </Field>

        <Field label="dataset_dir *" full>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input
                value={form.dataset_dir}
                onChange={(e) => update("dataset_dir", e.target.value)}
                disabled={disabled}
                className="form-input flex-[4] min-w-0"
                placeholder="/path/to/dataset"
              />
              <button
                type="button"
                onClick={openDatasetFolderPicker}
                disabled={uploadDisabled}
                className="px-3 py-2 rounded-md bg-muted text-foreground text-xs font-medium border border-border hover:bg-muted/80 disabled:opacity-40 whitespace-nowrap"
                title="Select folder and upload to backend dataset_raw/<exp>"
              >
                {uploadingDataset ? "Uploading..." : "Select Folder"}
              </button>
            </div>
            <input
              ref={datasetFolderInputRef}
              type="file"
              onChange={handleDatasetFilesChange}
              disabled={uploadDisabled}
              className="hidden"
              {...({ webkitdirectory: "true", directory: "" } as Record<string, string>)}
            />
            <p className="text-xs text-muted-foreground">
              Escribe la ruta manual o usa <code>Select Folder</code> para subir una carpeta a <code>dataset_raw/{form.exp || "<exp>"}</code>.
            </p>
          </div>
        </Field>

        <Field label="sr">
          <select value={form.sr} onChange={(e) => update("sr", e.target.value)} disabled={disabled} className="form-input">
            <option value="32k">32k</option>
            <option value="40k">40k</option>
            <option value="48k">48k</option>
          </select>
        </Field>
        <Field label="version">
          <select value={form.version} onChange={(e) => update("version", e.target.value)} disabled={disabled} className="form-input">
            <option value="v1">v1</option>
            <option value="v2">v2</option>
          </select>
        </Field>
        <Field label="device">
          <select value={form.device} onChange={(e) => update("device", e.target.value)} disabled={disabled} className="form-input">
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
          </select>
        </Field>
        <Field label="f0_method">
          <select value={form.f0_method} onChange={(e) => update("f0_method", e.target.value)} disabled={disabled} className="form-input">
            {["rmvpe_gpu", "rmvpe", "harvest", "pm", "dio"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </Field>
        <Field label="early_stop_metric">
          <select value={form.early_stop_metric} onChange={(e) => update("early_stop_metric", e.target.value)} disabled={disabled} className="form-input">
            <option value="loss_mel">loss_mel</option>
            <option value="loss_gen_all">loss_gen_all</option>
          </select>
        </Field>

        <Field label="if_f0">
          <Toggle checked={form.if_f0 === 1} onChange={(v) => update("if_f0", v ? 1 : 0)} disabled={disabled} />
        </Field>

        <Field label="np">
          <input type="number" value={form.np} onChange={(e) => update("np", +e.target.value)} disabled={disabled} className="form-input" min={1} />
        </Field>
        <Field label="gpus">
          <input value={form.gpus} onChange={(e) => update("gpus", e.target.value)} disabled={disabled} className="form-input" />
        </Field>
        <Field label="gpus_rmvpe">
          <input value={form.gpus_rmvpe} onChange={(e) => update("gpus_rmvpe", e.target.value)} disabled={disabled} className="form-input" />
        </Field>
        <Field label="batch_size">
          <input type="number" value={form.batch_size} onChange={(e) => update("batch_size", +e.target.value)} disabled={disabled} className="form-input" min={1} />
        </Field>
        <Field label="save_every_epoch">
          <input type="number" value={form.save_every_epoch} onChange={(e) => update("save_every_epoch", +e.target.value)} disabled={disabled} className="form-input" min={1} />
        </Field>
        <Field label="total_epoch">
          <input type="number" value={form.total_epoch} onChange={(e) => update("total_epoch", +e.target.value)} disabled={disabled} className="form-input" min={1} />
        </Field>
        <Field label="early_stop_patience">
          <input type="number" value={form.early_stop_patience} onChange={(e) => update("early_stop_patience", +e.target.value)} disabled={disabled} className="form-input" min={0} />
        </Field>
        <Field label="early_stop_min_delta">
          <input type="number" step="0.01" value={form.early_stop_min_delta} onChange={(e) => update("early_stop_min_delta", +e.target.value)} disabled={disabled} className="form-input" min={0} />
        </Field>

        <Field label="save_every_weights">
          <Toggle checked={form.save_every_weights} onChange={(v) => update("save_every_weights", v)} disabled={disabled} />
        </Field>
        <Field label="copy_to_models">
          <Toggle checked={form.copy_to_models} onChange={(v) => update("copy_to_models", v)} disabled={disabled} />
        </Field>
        <Field label="is_half">
          <Toggle checked={form.is_half} onChange={(v) => update("is_half", v)} disabled={disabled} />
        </Field>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={handleSubmit} disabled={disabled}
          className="px-6 py-2.5 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:opacity-90 disabled:opacity-40 transition-opacity">
          {loading ? "Starting…" : "Start Training"}
        </button>
        <button onClick={reset}
          className="px-6 py-2.5 rounded-md bg-muted text-muted-foreground font-medium text-sm hover:text-foreground border border-border transition-colors">
          Reset
        </button>
      </div>

      <JsonViewer data={response} />
    </div>
  );
};

// ── Sub-components ──

function Field({ label, children, full }: { label: string; children: React.ReactNode; full?: boolean }) {
  return (
    <label className={`block ${full ? "md:col-span-2" : ""}`}>
      <span className="text-xs font-mono text-muted-foreground mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onChange(!checked)}
      disabled={disabled}
      className={`w-10 h-5 rounded-full relative transition-colors ${checked ? "bg-primary" : "bg-muted border border-border"} disabled:opacity-40`}
    >
      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-foreground transition-transform ${checked ? "left-5" : "left-0.5"}`} />
    </button>
  );
}

export default TrainPanel;
