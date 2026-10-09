import { useEffect, useState } from "react";
import { FolderOpen, Save } from "lucide-react";
import { appLogDir } from "@tauri-apps/api/path";
import { openPath } from "@tauri-apps/plugin-opener";
import { Button, Card, CardHeader } from "../components/ui";
import { useToast } from "../components/Toast";
import { getHardwareId } from "../lib/hardware";
import { isTauri } from "../lib/tauri";
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type AppSettings } from "../lib/settings";

export function SettingsPage() {
  const toast = useToast();
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [logDir, setLogDir] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauri()) {
      return;
    }
    void appLogDir()
      .then(setLogDir)
      .catch(() => setLogDir(null));
  }, []);

  async function openLogs() {
    if (!logDir) {
      return;
    }
    try {
      await openPath(logDir);
    } catch {
      toast.push("error", "Could not open the log folder.");
    }
  }

  function update<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function save() {
    saveSettings(settings);
    toast.push("success", "Settings saved.");
  }

  function reset() {
    setSettings(DEFAULT_SETTINGS);
    saveSettings(DEFAULT_SETTINGS);
    toast.push("info", "Settings reset to defaults.");
  }

  return (
    <>
      <div className="mb-4 flex justify-end gap-2">
        <Button variant="secondary" onClick={reset}>
          Reset
        </Button>
        <Button icon={Save} onClick={save}>
          Save
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader title="API" subtitle="TokenMiner backend" />
          <div className="space-y-4 px-5 py-4">
            <TextField
              label="API base URL"
              value={settings.apiBaseUrl}
              onChange={(value) => update("apiBaseUrl", value)}
            />
            <TextField
              label="Stratum endpoint (proxy)"
              value={settings.stratumEndpoint}
              onChange={(value) => update("stratumEndpoint", value)}
            />
            <SelectField
              label="Default engine (sessions that name neither coin)"
              value={settings.defaultEngine}
              options={["pearl", "quantus"]}
              onChange={(value) => update("defaultEngine", value)}
            />
            <ReadOnlyField label="Hardware id" value={getHardwareId()} />
          </div>
        </Card>

        <Card>
          <CardHeader title="Pearl" subtitle="zk-pow proof engine" />
          <div className="space-y-4 px-5 py-4">
            <NumberField
              label="Certificate version"
              value={settings.pearl.certVersion}
              min={1}
              max={3}
              onChange={(value) => update("pearl", { ...settings.pearl, certVersion: value })}
            />
            {/* Read-only: the search runs entirely on the GPU, so there is no thread count to
                choose. The dimensions are fixed at what the pools price shares against, and the
                engine refuses anything else — shown here because a refused start names them. */}
            <ReadOnlyField label="Matrix dimensions" value="131072 x 262144 x 2048, rank 128" />
            <ReadOnlyField label="Hash tile" value="16 x 16" />
          </div>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader title="Quantus" subtitle="QUIC miner engine" />
          <div className="grid grid-cols-1 gap-4 px-5 py-4 md:grid-cols-2">
            <TextField
              label="Node address"
              value={settings.quantus.nodeAddr}
              onChange={(value) => update("quantus", { ...settings.quantus, nodeAddr: value })}
            />
            <NumberField
              label="Metrics port"
              value={settings.quantus.metricsPort}
              min={1}
              max={65535}
              onChange={(value) => update("quantus", { ...settings.quantus, metricsPort: value })}
            />
            <TextField
              label="Auth token file"
              value={settings.quantus.authTokenFile}
              onChange={(value) => update("quantus", { ...settings.quantus, authTokenFile: value })}
            />
            <TextField
              label="TLS cert SHA-256 file"
              value={settings.quantus.tlsCertSha256File}
              onChange={(value) => update("quantus", { ...settings.quantus, tlsCertSha256File: value })}
            />
            <NumberField
              label="CPU workers (0 = auto)"
              value={settings.quantus.cpuWorkers}
              min={0}
              max={128}
              onChange={(value) => update("quantus", { ...settings.quantus, cpuWorkers: value })}
            />
            <NumberField
              label="GPU devices (0 = auto)"
              value={settings.quantus.gpuDevices}
              min={0}
              max={16}
              onChange={(value) => update("quantus", { ...settings.quantus, gpuDevices: value })}
            />
            <ToggleField
              label="Use native CUDA (NVIDIA)"
              value={settings.quantus.cudaGpu}
              onChange={(value) => update("quantus", { ...settings.quantus, cudaGpu: value })}
            />
            <ToggleField
              label="Allow integrated GPU"
              value={settings.quantus.allowIntegrated}
              onChange={(value) => update("quantus", { ...settings.quantus, allowIntegrated: value })}
            />
          </div>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader title="Diagnostics" subtitle="Where the Rust log file lives, for debugging in dev." />
          <div className="flex flex-wrap items-end gap-3 px-5 py-4">
            <div className="min-w-64 flex-1">
              <ReadOnlyField label="Log folder" value={logDir ?? "unavailable"} />
            </div>
            <Button variant="secondary" icon={FolderOpen} onClick={() => void openLogs()} disabled={!logDir}>
              Open logs
            </Button>
          </div>
        </Card>
      </div>
    </>
  );
}

const INPUT_CLASS =
  "h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 focus:border-brand-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100";

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">{label}</span>
      <input className={INPUT_CLASS} value={value} onChange={(e) => onChange(e.currentTarget.value)} />
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">{label}</span>
      <input
        type="number"
        className={INPUT_CLASS}
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
      />
    </label>
  );
}

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">{label}</span>
      <input readOnly className={`${INPUT_CLASS} font-mono text-xs`} value={value} />
    </label>
  );
}

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">{label}</span>
      <select className={INPUT_CLASS} value={value} onChange={(e) => onChange(e.currentTarget.value as T)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function ToggleField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-4 py-2">
      <span className="text-sm text-slate-600 dark:text-slate-300">{label}</span>
      <input
        type="checkbox"
        className="h-4 w-4 accent-brand-600"
        checked={value}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
    </label>
  );
}
