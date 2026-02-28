"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { startConvert, getJobStatus, downloadJobOutput, listModels, type JobStatus } from "@/lib/api";
import JsonViewer from "@/components/JsonViewer";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const STORAGE_KEY = "rvc_convert_form";

interface ConvertForm {
  model: string;
  index: string;
  auto_transpose: boolean;
  index_rate: number;
  protect: number;
  rms_mix_rate: number;
}

const defaultValues: ConvertForm = {
  model: "",
  index: "",
  auto_transpose: true,
  index_rate: 0.75,
  protect: 0.33,
  rms_mix_rate: 0.25,
};

function loadSaved(): ConvertForm {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s) return { ...defaultValues, ...JSON.parse(s) };
  } catch {}
  return { ...defaultValues };
}

const ConvertPanel = () => {
  const [form, setForm] = useState<ConvertForm>(loadSaved);
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [polling, setPolling] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [indexes, setIndexes] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
  }, [form]);

  useEffect(() => {
    const loadModels = async () => {
      setLoadingModels(true);
      try {
        const data = await listModels();
        const modelList = data.pth || data.models || [];
        const indexList = data.index || data.indexes || [];
        setModels(modelList);
        setIndexes(indexList);
        if (modelList.length === 0) {
          setError("No models available. Make sure models are uploaded to the backend.");
        }
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : "Failed to load models";
        console.error("Failed to load models:", err);
        setError(`Error loading models: ${errorMsg}`);
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setPolling(false);
  }, []);

  const startPolling = useCallback((id: string) => {
    setPolling(true);
    let failCount = 0;
    pollingRef.current = setInterval(async () => {
      try {
        const st = await getJobStatus(id);
        setJobStatus(st);
        setResponse(st);
        failCount = 0; // Reset fail count on success
        if (st.status === "completed" || st.status === "failed") {
          stopPolling();
          if (st.status === "failed") {
            setError(st.error || "Conversion failed");
            setStatus("failed");
          } else {
            setStatus("completed");
          }
        }
      } catch (err: unknown) {
        failCount++;
        const errorMsg = err instanceof Error ? err.message : "Poll error";
        console.error("Poll error:", err);
        if (failCount > 3) {
          stopPolling();
          setError(`Failed to check job status: ${errorMsg}`);
          setStatus("failed");
        }
      }
    }, 2000);
  }, [stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const handleSubmit = async () => {
    // Validate inputs
    if (!inputFile) {
      setError("Input voice file is required");
      return;
    }
    if (inputFile.type !== "audio/wav" && !inputFile.name.endsWith(".wav")) {
      setError("Only WAV files are supported");
      return;
    }
    if (!form.model.trim()) {
      setError("Model selection is required");
      return;
    }
    setError(null);
    setStatus("uploading");
    setLoading(true);
    setResponse(null);
    setJobId(null);
    setJobStatus(null);
    abortRef.current = new AbortController();

    const fd = new FormData();
    fd.append("file", inputFile);
    if (refFile) fd.append("reference_file", refFile);
    fd.append("model", form.model);
    if (form.index.trim()) fd.append("index", form.index);
    fd.append("auto_transpose", String(form.auto_transpose));
    fd.append("index_rate", String(form.index_rate));
    fd.append("protect", String(form.protect));
    fd.append("rms_mix_rate", String(form.rms_mix_rate));

    try {
      const res = await startConvert(fd, abortRef.current.signal);
      const contentType = res.headers.get("content-type") || "";

      try {
        if (contentType.includes("audio") || contentType.includes("octet-stream")) {
          // Direct audio blob response
          const blob = await res.blob();
          triggerDownload(blob, `converted_${inputFile.name}`);
          setStatus("success");
          setResponse({ message: "Conversion complete, file downloaded." });
        } else {
          // JSON response - could be job-based
          const json = await res.json();
          setResponse(json);

          if (json.job_id) {
            setJobId(json.job_id);
            setStatus("converting");
            startPolling(json.job_id);
          } else if (json.url) {
            triggerDownloadUrl(json.url, `converted_${inputFile.name}`);
            setStatus("success");
          } else if (json.audio_base64) {
            const binary = atob(json.audio_base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const blob = new Blob([bytes], { type: "audio/wav" });
            triggerDownload(blob, `converted_${inputFile.name}`);
            setStatus("success");
          } else {
            setStatus("success");
          }
        }
      } catch (parseErr: unknown) {
        const errorMsg = parseErr instanceof Error ? parseErr.message : "Failed to parse response";
        throw new Error(`Response parsing error: ${errorMsg}`);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.name === "AbortError") {
          setStatus("aborted");
          setError("Conversion cancelled");
        } else {
          setError(err.message);
          setStatus("failed");
        }
      } else {
        setError("An unexpected error occurred");
        setStatus("failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!jobId) {
      setError("No job ID available");
      return;
    }
    try {
      const blob = await downloadJobOutput(jobId);
      if (blob.size === 0) {
        setError("Downloaded file is empty");
        return;
      }
      triggerDownload(blob, `converted_${inputFile?.name || "output.wav"}`);
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Download failed";
      setError(errorMsg);
    }
  };

  const reset = () => {
    stopPolling();
    setForm({ ...defaultValues });
    setInputFile(null);
    setRefFile(null);
    setJobId(null);
    setJobStatus(null);
    setError(null);
    setResponse(null);
    setStatus(null);
    setLoading(false);
    localStorage.removeItem(STORAGE_KEY);
  };

  const update = <K extends keyof ConvertForm>(key: K, val: ConvertForm[K]) => {
    setForm((f) => ({ ...f, [key]: val }));
  };

  const disabled = loading || polling;

  const statusColor = (s?: string) => {
    switch (s) {
      case "completed": case "success": return "text-success";
      case "failed": return "text-destructive";
      case "converting": case "running": return "text-warning";
      default: return "text-info";
    }
  };

  return (
    <div className="space-y-6">
      {/* Status */}
      {(status || jobId) && (
        <div className="bg-secondary rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              {jobId && (
                <>
                  <span className="text-xs font-mono text-muted-foreground">JOB</span>
                  <code className="text-sm font-mono text-foreground">{jobId}</code>
                </>
              )}
              <span className={`text-sm font-mono font-semibold uppercase ${statusColor(jobStatus?.status || status || "")}`}>
                {polling && <span className="inline-block w-2 h-2 rounded-full bg-current mr-2 animate-pulse-dot" />}
                {jobStatus?.status || status}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {polling && (
                <button onClick={stopPolling} className="text-xs px-3 py-1 rounded bg-muted text-muted-foreground hover:text-foreground border border-border transition-colors">
                  Stop Polling
                </button>
              )}
              {(jobStatus?.status === "completed") && (
                <button onClick={handleDownload} className="text-xs px-3 py-1 rounded bg-primary text-primary-foreground hover:opacity-90 transition-opacity">
                  Download Result
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FileField label="Input Voice (wav) *" accept=".wav" file={inputFile} onChange={setInputFile} disabled={disabled} />
        <FileField label="Reference Voice (wav)" accept=".wav" file={refFile} onChange={setRefFile} disabled={disabled} />

        <Field  label="model *">
          <Select value={form.model} onValueChange={(v) => update("model", v)} disabled={disabled || loadingModels}>
            <SelectTrigger>
              <SelectValue placeholder={loadingModels ? "Loading models..." : "Select a model"} />
            </SelectTrigger>
            <SelectContent>
              {models.map((m) => (
                <SelectItem  key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="index">
          <div className="flex gap-2 items-center">
            <Select value={form.index} onValueChange={(v) => update("index", v)} disabled={disabled || loadingModels}>
              <SelectTrigger>
                <SelectValue placeholder="Select an index (optional)" />
              </SelectTrigger>
              <SelectContent>
                {indexes.map((idx) => (
                  <SelectItem key={idx} value={idx}>
                    {idx}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {form.index && (
              <button
                onClick={() => update("index", "")}
                disabled={disabled}
                className="text-xs px-2 py-1 rounded bg-muted text-muted-foreground hover:text-foreground border border-border transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </Field>

        <Field label="auto_transpose">
          <Toggle checked={form.auto_transpose} onChange={(v) => update("auto_transpose", v)} disabled={disabled} />
        </Field>

        <SliderField label="index_rate" value={form.index_rate} onChange={(v) => update("index_rate", v)} disabled={disabled} />
        <SliderField label="protect" value={form.protect} onChange={(v) => update("protect", v)} disabled={disabled} />
        <SliderField label="rms_mix_rate" value={form.rms_mix_rate} onChange={(v) => update("rms_mix_rate", v)} disabled={disabled} />
      </div>

      <div className="flex gap-3">
        <button onClick={handleSubmit} disabled={disabled}
          className="px-6 py-2.5 rounded-md bg-primary text-primary-foreground font-medium text-sm hover:opacity-90 disabled:opacity-40 transition-opacity">
          {loading ? "Converting…" : "Convert"}
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

// ── Helpers ──

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function triggerDownloadUrl(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-mono text-muted-foreground mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function FileField({ label, accept, file, onChange, disabled }: {
  label: string; accept: string; file: File | null; onChange: (f: File | null) => void; disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-xs font-mono text-muted-foreground mb-1 block">{label}</span>
      <div className="relative">
        <input type="file" accept={accept} disabled={disabled}
          onChange={(e) => onChange(e.target.files?.[0] || null)}
          className="form-input file:mr-3 file:px-3 file:py-1 file:rounded file:border-0 file:bg-primary file:text-primary-foreground file:text-xs file:font-medium file:cursor-pointer" />
        {file && <span className="text-xs text-muted-foreground mt-1 block">{file.name}</span>}
      </div>
    </label>
  );
}

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button type="button" onClick={() => !disabled && onChange(!checked)} disabled={disabled}
      className={`w-10 h-5 rounded-full relative transition-colors ${checked ? "bg-primary" : "bg-muted border border-border"} disabled:opacity-40`}>
      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-foreground transition-transform ${checked ? "left-5" : "left-0.5"}`} />
    </button>
  );
}

function SliderField({ label, value, onChange, disabled }: {
  label: string; value: number; onChange: (v: number) => void; disabled?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-xs font-mono text-muted-foreground mb-1 block">{label}: {value.toFixed(2)}</span>
      <input type="range" min={0} max={1} step={0.01} value={value}
        onChange={(e) => onChange(+e.target.value)} disabled={disabled}
        className="w-full h-1.5 rounded-full appearance-none bg-muted accent-primary cursor-pointer disabled:opacity-40" />
    </label>
  );
}

export default ConvertPanel;
