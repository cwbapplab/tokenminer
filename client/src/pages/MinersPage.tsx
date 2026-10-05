import { useEffect, useState } from "react";
import { Coins, Cpu, Hash, Play, Server, Square, Terminal } from "lucide-react";
import { Badge, Button, Card, CardHeader } from "../components/ui";
import { LogConsole, type LogLine } from "../components/LogConsole";
import { useMiners } from "../hooks/useMiners";
import { useBackendLogs } from "../hooks/useBackendLogs";
import { useToast } from "../components/Toast";
import { onMinerLog, startSessionMiner, stopMiner } from "../lib/miners";
import { startSession, stopSession, getSession } from "../lib/session";
import { startHeartbeat, stopHeartbeat } from "../lib/heartbeat";
import { loadSettings } from "../lib/settings";
import { formatHashrate, formatPearlRate } from "../lib/utils";
import type { MinerCommandParams, MinerKind, MinerStatus, MiningSession } from "../lib/types";

const ENGINE_CARDS: Array<{ kind: MinerKind; name: string; coin: string; blurb: string }> = [
  {
    kind: "quantus",
    name: "Quantus",
    coin: "QTC",
    blurb: "Native CPU / GPU / CUDA engine, driven in-process.",
  },
  {
    kind: "pearl",
    name: "Pearl",
    coin: "PRL",
    blurb: "Native zk-pow proof engine, driven in-process.",
  },
];

const STATE_TONE: Record<MinerStatus["state"], "neutral" | "brand" | "success" | "danger"> = {
  stopped: "neutral",
  starting: "brand",
  running: "success",
  error: "danger",
};

export function MinersPage() {
  const miners = useMiners();
  const backendLogs = useBackendLogs();
  const toast = useToast();
  const [logs, setLogs] = useState<Record<MinerKind, LogLine[]>>({ quantus: [], pearl: [] });
  const [session, setSession] = useState<MiningSession | null>(null);
  const [engine, setEngine] = useState<MinerKind | null>(null);
  const [params, setParams] = useState<MinerCommandParams | null>(null);
  const [pending, setPending] = useState<MinerKind | null>(null);

  useEffect(() => {
    let disposed = false;
    let detach: (() => void) | undefined;

    void (async () => {
      const off = await onMinerLog((line) => {
        if (disposed) {
          return;
        }
        setLogs((current) => {
          const entry: LogLine = {
            id: Date.now() + Math.random(),
            level: line.level,
            message: line.message,
            timestamp: line.timestamp,
          };
          return { ...current, [line.kind]: [...current[line.kind], entry].slice(-200) };
        });
      });

      if (disposed) {
        off();
      } else {
        detach = off;
      }
    })();

    return () => {
      disposed = true;
      detach?.();
      stopHeartbeat();
    };
  }, []);

  const sessionRunning =
    engine !== null && (miners[engine].state === "running" || miners[engine].state === "starting");

  /** Runs the engine for a session, whether freshly started or adopted. */
  async function runSession(created: MiningSession): Promise<MinerKind> {
    const settings = loadSettings();
    const started = await startSessionMiner({
      command: created.minerCommand,
      minerConfig: created.minerConfig,
      coinCode: created.coinCode,
      algorithmCode: created.algorithmCode,
      // The API's endpoint is authoritative; the local setting is only a dev fallback.
      stratumEndpoint: created.stratumEndpoint || settings.stratumEndpoint,
      quantus: settings.quantus,
      pearl: settings.pearl,
    });
    setSession(created);
    setEngine(started.kind);
    setParams(started.params);
    startHeartbeat();
    return started.kind;
  }

  // The API may already hold a session for this device (for example, the app restarted while
  // mining). Adopt it so the UI reflects reality instead of being blocked by "already active".
  useEffect(() => {
    let disposed = false;

    void (async () => {
      try {
        const existing = await getSession();
        if (!existing || disposed) {
          return;
        }
        await runSession(existing);
      } catch {
        // Not signed in, offline, or no session — nothing to adopt.
      }
    })();

    return () => {
      disposed = true;
    };
    // Once per visit; runSession/startSessionMiner are idempotent.
  }, []);

  /** Opens an API session, parses its command in Rust, and runs the mapped engine. */
  async function toggleSession(running: boolean) {
    setPending(engine ?? "quantus");
    try {
      if (running && engine) {
        await stopMiner(engine);
        stopHeartbeat();
        try {
          await stopSession("user-triggered");
        } catch {
          /* The local stop matters more than the API round-trip. */
        }
        setSession(null);
        setEngine(null);
        setParams(null);
        toast.push("info", "Mining session stopped.");
      } else {
        let created: MiningSession | null = null;
        try {
          created = await startSession();
        } catch (error) {
          // A session is already active for this device: adopt it rather than failing.
          created = await getSession().catch(() => null);
          if (!created) {
            throw error;
          }
          toast.push("info", "Adopted the session the API already had for this device.");
        }

        const kind = await runSession(created);
        toast.push("success", `Mining ${created.coinCode} on ${created.poolName} (${kind}).`);
      }
    } catch (error) {
      toast.push("error", error instanceof Error ? error.message : "Could not start the mining session.");
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader
          title="Mining session"
          subtitle="Asks the API for a session, parses its launch command, and runs the matching engine in this process."
          action={
            <Badge tone={STATE_TONE[engine ? miners[engine].state : "stopped"]}>
              {engine ? `${engine} · ${miners[engine].state}` : "idle"}
            </Badge>
          }
        />

        {session ? (
          <div className="grid grid-cols-1 gap-3 px-5 py-4 sm:grid-cols-3">
            <Metric icon={Server} label="Pool" value={session.poolName} />
            <Metric icon={Coins} label="Coin" value={`${session.coinCode} · ${session.algorithmCode}`} />
            <Metric icon={Hash} label="Worker" value={session.workerId} mono />
          </div>
        ) : (
          <p className="px-5 py-4 text-sm text-slate-500 dark:text-slate-400">
            No active session. Starting one opens a session, then the Rust side parses the returned command and
            drives the embedded engine with it.
          </p>
        )}

        {session ? (
          <div className="grid grid-cols-1 gap-4 px-5 pb-4 lg:grid-cols-2">
            <div>
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Command</p>
              <pre className="scrollbar-slim overflow-x-auto rounded-md bg-slate-950 p-3 text-xs text-slate-300">
                {session.minerCommand}
              </pre>
            </div>
            <div>
              <p className="mb-1 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500">
                <Cpu className="h-3.5 w-3.5" /> Parsed parameters
              </p>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <ParsedParam label="program" value={params?.program} />
                <ParsedParam label="algo" value={params?.algo} />
                <ParsedParam label="host" value={params?.host} />
                <ParsedParam label="wallet" value={params?.wallet} />
                <ParsedParam label="worker" value={params?.worker} />
                <ParsedParam label="cpu / gpu" value={formatWorkers(params)} />
              </dl>
            </div>
          </div>
        ) : null}

        <div className="flex gap-2 border-t border-slate-100 px-5 py-4 dark:border-slate-800">
          <Button
            variant={sessionRunning ? "danger" : "primary"}
            icon={sessionRunning ? Square : Play}
            loading={pending !== null && sessionRunning}
            onClick={() => void toggleSession(sessionRunning)}
          >
            {sessionRunning ? "Stop mining" : "Start mining"}
          </Button>
        </div>

        <div className="border-t border-slate-100 px-5 py-4 dark:border-slate-800">
          <p className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Terminal className="h-3.5 w-3.5" /> Miner output
          </p>
          <LogConsole lines={engine ? logs[engine] : []} />
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {ENGINE_CARDS.map((card) => {
          const status = miners[card.kind];
          // Each engine's status carries its rate in its own unit — Quantus H/s, Pearl TH/s — so the
          // formatter is chosen per card. `formatHashrate` on a Pearl rate would rescale 56 TH/s into
          // "56.00 KH/s".
          const formatRate = card.kind === "pearl" ? formatPearlRate : formatHashrate;
          return (
            <Card key={card.kind} className="flex flex-col">
              <CardHeader
                title={
                  <span className="flex items-center gap-2">
                    {card.name}
                    <span className="text-xs font-normal text-slate-400">{card.coin}</span>
                  </span>
                }
                subtitle={card.blurb}
                action={<Badge tone={STATE_TONE[status.state]}>{status.state}</Badge>}
              />

              <div className="grid grid-cols-3 gap-3 px-5 py-4">
                <Metric label="Hashrate" value={formatRate(status.hashrate)} />
                <Metric label="Workers" value={String(status.workers)} />
                <Metric label="Jobs" value={String(status.activeJobs)} />
              </div>

              {status.message ? (
                <p className="px-5 pb-3 text-xs text-slate-500 dark:text-slate-400">{status.message}</p>
              ) : null}

              <div className="border-t border-slate-100 px-5 py-4 dark:border-slate-800">
                <p className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  <Terminal className="h-3.5 w-3.5" /> Log
                </p>
                <LogConsole lines={logs[card.kind]} />
              </div>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader
          title="Backend log"
          subtitle="Rust log output (miner-service, zk-pow, plugins). File: see Settings → Diagnostics."
        />
        <div className="px-5 py-4">
          <LogConsole lines={backendLogs} className="h-80" />
        </div>
      </Card>
    </div>
  );
}

function formatWorkers(params: MinerCommandParams | null): string | null {
  if (!params) {
    return null;
  }
  const cpu = params.cpuWorkers ?? "auto";
  const gpu = params.gpuDevices ?? "auto";
  return `${cpu} / ${gpu}`;
}

function ParsedParam({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <>
      <dt className="text-slate-400">{label}</dt>
      <dd className="truncate font-mono text-slate-700 dark:text-slate-200" title={value ?? undefined}>
        {value ?? "—"}
      </dd>
    </>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  mono = false,
}: {
  icon?: typeof Server;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/60">
      <p className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
        {Icon ? <Icon className="h-3.5 w-3.5" /> : null}
        {label}
      </p>
      <p
        className={`mt-0.5 truncate text-sm font-semibold text-slate-800 dark:text-slate-100 ${mono ? "font-mono text-xs" : ""}`}
        title={value}
      >
        {value}
      </p>
    </div>
  );
}
