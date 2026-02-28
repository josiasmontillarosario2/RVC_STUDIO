const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  "";

const API_PREFIX = `${BASE_URL}/api/rvc`;

// ── Types ──

export interface TrainRequest {
  dataset_dir: string;
  exp: string;
  sr: string;
  version: string;
  if_f0: number;
  np: number;
  f0_method: string;
  gpus: string;
  gpus_rmvpe: string;
  batch_size: number;
  save_every_epoch: number;
  total_epoch: number;
  early_stop_patience: number;
  early_stop_min_delta: number;
  early_stop_metric: string;
  save_every_weights: boolean;
  copy_to_models: boolean;
  device: string;
  is_half: boolean;
}

export interface JobResponse {
  job_id: string;
  status: string;
  kind: string;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  kind: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  result?: {
    output_path?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface ModelsListResponse {
  pth?: string[];
  index?: string[];
  models?: string[];
  indexes?: string[];
}

export interface DatasetsListResponse {
  datasets: string[];
}

export interface UploadDatasetResponse {
  message: string;
  exp: string;
  dataset_dir: string;
  files_saved: string[];
  count: number;
}

export interface JobRecord {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  kind: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  command?: string;
  result?: Record<string, unknown>;
  log_tail?: string;
  [key: string]: unknown;
}

export interface JobsListResponse {
  jobs: JobRecord[];
}

export interface HealthResponse {
  status: string;
  repo_root?: string;
  [key: string]: unknown;
}

// ── Helpers ──

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    let detail: string;
    try {
      const json = JSON.parse(text);
      detail = json.detail || json.message || JSON.stringify(json);
    } catch {
      detail = text || res.statusText || `HTTP ${res.status}`;
    }
    throw new Error(detail);
  }
  try {
    return res.json();
  } catch (err) {
    throw new Error("Invalid JSON response from server");
  }
}

// ── API calls ──

export async function startTrain(body: TrainRequest, signal?: AbortSignal): Promise<JobResponse> {
  const res = await fetch(`${API_PREFIX}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return handleResponse<JobResponse>(res);
}

export async function getJobStatus(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
  const res = await fetch(`${API_PREFIX}/jobs/${jobId}`, { signal });
  return handleResponse<JobStatus>(res);
}

export async function startConvert(formData: FormData, signal?: AbortSignal): Promise<Response> {
  // Returns raw response so caller can handle blob vs json
  const res = await fetch(`${API_PREFIX}/convert`, {
    method: "POST",
    body: formData,
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    let detail: string;
    try {
      const json = JSON.parse(text);
      detail = json.detail || json.message || JSON.stringify(json);
    } catch {
      detail = text || res.statusText || `HTTP ${res.status}`;
    }
    throw new Error(detail);
  }
  return res;
}

export async function downloadJobOutput(jobId: string, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch(`${API_PREFIX}/jobs/${jobId}/download`, { signal });
  if (!res.ok) {
    const text = await res.text();
    let detail: string;
    try {
      const json = JSON.parse(text);
      detail = json.detail || json.message || JSON.stringify(json);
    } catch {
      detail = text || res.statusText || `HTTP ${res.status}`;
    }
    throw new Error(`Failed to download: ${detail}`);
  }
  return res.blob();
}

export async function listModels(signal?: AbortSignal): Promise<ModelsListResponse> {
  const res = await fetch(`${API_PREFIX}/models`, { signal });
  return handleResponse<ModelsListResponse>(res);
}

export async function listDatasets(signal?: AbortSignal): Promise<DatasetsListResponse> {
  const res = await fetch(`${API_PREFIX}/datasets`, { signal });
  return handleResponse<DatasetsListResponse>(res);
}

export async function uploadDataset(
  exp: string,
  files: File[],
  clearExisting: boolean = true,
  signal?: AbortSignal
): Promise<UploadDatasetResponse> {
  const formData = new FormData();
  formData.append("exp", exp);
  formData.append("clear_existing", String(clearExisting));
  files.forEach((file) => {
    formData.append("files", file);
  });
  
  const res = await fetch(`${API_PREFIX}/uploads/dataset`, {
    method: "POST",
    body: formData,
    signal,
  });
  return handleResponse<UploadDatasetResponse>(res);
}

export async function listJobs(signal?: AbortSignal): Promise<JobsListResponse> {
  const res = await fetch(`${API_PREFIX}/jobs`, { signal });
  return handleResponse<JobsListResponse>(res);
}

export async function getJobLogs(jobId: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(`${API_PREFIX}/jobs/${jobId}/logs`, { signal });
  if (!res.ok) {
    const text = await res.text();
    try {
      const json = JSON.parse(text);
      const detail = json.detail || json.message || JSON.stringify(json);
      throw new Error(detail);
    } catch (err) {
      if (err instanceof Error && err.message.includes("JSON")) {
        throw new Error(text || `HTTP ${res.status}: Failed to fetch logs`);
      }
      throw err;
    }
  }
  return res.text();
}

export async function healthCheck(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_PREFIX}/health`, { signal });
  return handleResponse<HealthResponse>(res);
}
