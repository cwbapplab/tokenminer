//! Parses the launch command the API renders and extracts the parameters the
//! embedded engines can act on.
//!
//! We deliberately do **not** execute the command as a process. The template is a
//! Stratum-miner command line (`rgminer.exe --algo {algo} --stratum {stratumHost}
//! --wallet {wallet} --worker-name {workerId}`), so we tokenize it and pull out the
//! values we need, then drive the in-process engines with them.

use serde::Serialize;

use super::MinerKind;

/// The parameters we can recover from a rendered miner command.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MinerParams {
    pub program: String,
    pub algo: Option<String>,
    /// Stratum endpoint (`--stratum`, `--pool`, `-o`, …).
    pub host: Option<String>,
    pub wallet: Option<String>,
    pub worker: Option<String>,
    pub cpu_workers: Option<u32>,
    pub gpu_devices: Option<u32>,
    /// The raw tokens, for display.
    pub args: Vec<String>,
}

impl MinerParams {
    pub fn describe(&self) -> String {
        let mut parts = vec![format!("program={}", self.program)];
        if let Some(algo) = &self.algo {
            parts.push(format!("algo={algo}"));
        }
        if let Some(host) = &self.host {
            parts.push(format!("host={host}"));
        }
        if let Some(wallet) = &self.wallet {
            parts.push(format!("wallet={wallet}"));
        }
        if let Some(worker) = &self.worker {
            parts.push(format!("worker={worker}"));
        }
        parts.join(" ")
    }
}

struct Flag {
    name: String,
    value: Option<String>,
}

/// Splits a command line into tokens, honouring single/double quotes.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else if c == '\\' && q == '"' {
                    if let Some(&next) = chars.peek() {
                        current.push(next);
                        chars.next();
                    }
                } else {
                    current.push(c);
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                } else if c.is_whitespace() {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push(c);
                }
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn normalize(name: &str) -> String {
    name.trim_start_matches('-').to_ascii_lowercase()
}

/// Collects `--flag value`, `--flag=value` and short `-x value` pairs, plus any
/// positional (non-flag) tokens.
fn collect_flags(tokens: &[String]) -> (Vec<Flag>, Vec<String>) {
    let mut flags = Vec::new();
    let mut positionals = Vec::new();
    let mut index = 0;

    while index < tokens.len() {
        let token = &tokens[index];

        if token.starts_with('-') && token != "-" {
            if let Some((name, value)) = token.split_once('=') {
                flags.push(Flag {
                    name: normalize(name),
                    value: Some(value.to_string()),
                });
            } else {
                let name = normalize(token);
                let next = tokens.get(index + 1);
                // A following token is this flag's value unless it is another flag.
                if let Some(next) = next {
                    if !next.starts_with('-') {
                        flags.push(Flag {
                            name,
                            value: Some(next.clone()),
                        });
                        index += 1;
                    } else {
                        flags.push(Flag { name, value: None });
                    }
                } else {
                    flags.push(Flag { name, value: None });
                }
            }
        } else {
            positionals.push(token.clone());
        }

        index += 1;
    }

    (flags, positionals)
}

/// Strips a `stratum+tcp://`-style scheme so the value is a plain `host:port`.
fn normalize_host(value: &str) -> String {
    const SCHEMES: [&str; 8] = [
        "stratum+tcp://",
        "stratum+ssl://",
        "stratum://",
        "tcp://",
        "ssl://",
        "http://",
        "https://",
        "ws://",
    ];

    let lower = value.to_ascii_lowercase();
    for scheme in SCHEMES {
        if lower.starts_with(scheme) {
            return value[scheme.len()..].to_string();
        }
    }

    value.to_string()
}

/// True for a bare `host:port` token.
fn looks_like_host(value: &str) -> bool {
    let value = normalize_host(value);
    match value.rsplit_once(':') {
        Some((host, port)) => !host.is_empty() && port.parse::<u16>().is_ok(),
        None => false,
    }
}

fn find<'a>(flags: &'a [Flag], aliases: &[&str]) -> Option<&'a str> {
    let aliases: Vec<String> = aliases.iter().map(|a| normalize(a)).collect();
    flags
        .iter()
        .find(|flag| aliases.iter().any(|alias| alias == &flag.name))
        .and_then(|flag| flag.value.as_deref())
}

fn find_u32(flags: &[Flag], aliases: &[&str]) -> Option<u32> {
    find(flags, aliases).and_then(|value| value.trim().parse().ok())
}

/// Parses a rendered miner command into [`MinerParams`].
///
/// A structured algorithm configuration has no command string at all, so a blank one parses to
/// empty params — the caller then relies on the config fields it received separately.
pub fn parse(command: &str) -> Result<MinerParams, String> {
    let tokens = tokenize(command);
    let Some(program) = tokens.first().cloned() else {
        return Ok(MinerParams::default());
    };

    let (flags, positionals) = collect_flags(&tokens[1..]);

    let algo = find(&flags, &["algo", "-a"]).map(str::to_string);
    let mut host = find(
        &flags,
        &[
            "stratum",
            "stratum-url",
            "stratumurl",
            "pool",
            "pool-url",
            "poolurl",
            "url",
            "host",
            "-o",
            "server",
        ],
    )
    .map(normalize_host);

    // Some templates put the endpoint positionally (e.g. `miner host:3333 wallet`).
    if host.is_none() {
        host = positionals
            .iter()
            .find(|token| looks_like_host(token))
            .map(|token| normalize_host(token));
    }

    let mut wallet =
        find(&flags, &["wallet", "user", "--user", "-u", "address"]).map(str::to_string);
    let mut worker = find(
        &flags,
        &["worker-name", "workername", "worker", "-w", "rig"],
    )
    .map(str::to_string);

    // `-u wallet.worker` is the common CPUMiner-style combined form.
    if worker.is_none() {
        if let Some(combined) = wallet.clone() {
            if let Some((w, rest)) = combined.split_once('.') {
                wallet = Some(w.to_string());
                worker = Some(rest.to_string());
            }
        }
    }

    Ok(MinerParams {
        program,
        algo,
        host,
        wallet,
        worker,
        cpu_workers: find_u32(&flags, &["cpu-workers", "cpuworkers", "threads", "-t"]),
        gpu_devices: find_u32(&flags, &["gpu-devices", "gpudevices", "gpus"]),
        args: tokens,
    })
}

/// Chooses the embedded engine for a session from the coin/algorithm, falling back
/// to the program name (e.g. `...pearl...` → Pearl), and finally to `default`.
///
/// Both coins have to be recognised explicitly now that the last branch is a preference rather than
/// a constant. `qtc` used to reach Quantus only because Quantus *was* the fallback, so a Pearl
/// default with only Pearl listed would send every Quantus session to the Pearl engine the moment the
/// default changed. The explicit sets are what keep a session on the engine its coin names.
pub fn resolve_kind(
    coin_code: Option<&str>,
    algorithm_code: Option<&str>,
    program: &str,
    default: MinerKind,
) -> MinerKind {
    let haystack = format!(
        "{} {} {}",
        coin_code.unwrap_or_default(),
        algorithm_code.unwrap_or_default(),
        program
    )
    .to_ascii_lowercase();

    if haystack.contains("prl") || haystack.contains("pearl") {
        MinerKind::Pearl
    } else if haystack.contains("qtc") || haystack.contains("quantus") {
        MinerKind::Quantus
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_template_shape() {
        let params = parse(
            "rgminer.exe --algo qtc --stratum localhost:3333 --wallet wallet123 --worker-name abcdef",
        )
        .unwrap();

        assert_eq!(params.program, "rgminer.exe");
        assert_eq!(params.algo.as_deref(), Some("qtc"));
        assert_eq!(params.host.as_deref(), Some("localhost:3333"));
        assert_eq!(params.wallet.as_deref(), Some("wallet123"));
        assert_eq!(params.worker.as_deref(), Some("abcdef"));
    }

    #[test]
    fn splits_combined_user() {
        let params = parse("miner -o pool:3333 -u wallet.rig7").unwrap();
        assert_eq!(params.wallet.as_deref(), Some("wallet"));
        assert_eq!(params.worker.as_deref(), Some("rig7"));
        assert_eq!(params.host.as_deref(), Some("pool:3333"));
    }

    #[test]
    fn handles_equals_form() {
        let params = parse("miner --stratum=localhost:3333 --algo=prl").unwrap();
        assert_eq!(params.host.as_deref(), Some("localhost:3333"));
        assert_eq!(params.algo.as_deref(), Some("prl"));
    }

    #[test]
    fn resolves_kind_from_coin_and_program() {
        // Every case here names an engine, so each still resolves the same way whatever the default
        // is -- that is the point of passing a default that disagrees with the answer.
        assert_eq!(
            resolve_kind(Some("PRL"), None, "miner", MinerKind::Quantus),
            MinerKind::Pearl
        );
        assert_eq!(
            resolve_kind(None, Some("quantus"), "miner", MinerKind::Pearl),
            MinerKind::Quantus
        );
        assert_eq!(
            resolve_kind(None, None, "pearl-miner", MinerKind::Quantus),
            MinerKind::Pearl
        );
        // `qtc` is the case the new default would have broken: it names the coin, not the engine, and
        // used to resolve only through the fallback.
        assert_eq!(
            resolve_kind(Some("qtc"), None, "miner", MinerKind::Pearl),
            MinerKind::Quantus
        );
    }

    #[test]
    fn ambiguous_sessions_use_the_default() {
        assert_eq!(
            resolve_kind(Some("XYZ"), Some("xyz"), "miner", MinerKind::Pearl),
            MinerKind::Pearl
        );
        assert_eq!(
            resolve_kind(Some("XYZ"), Some("xyz"), "miner", MinerKind::Quantus),
            MinerKind::Quantus
        );
    }

    #[test]
    fn strips_a_stratum_scheme_from_the_host() {
        let params = parse("peakminer --url stratum+tcp://localhost:3333 --wallet w1").unwrap();
        assert_eq!(params.host.as_deref(), Some("localhost:3333"));

        let params = parse("peakminer --pool https://pool.kryptex.com --wallet w1").unwrap();
        assert_eq!(params.host.as_deref(), Some("pool.kryptex.com"));
    }

    #[test]
    fn finds_a_positional_host() {
        let params = parse("peakminer localhost:3333 w1 worker1").unwrap();
        assert_eq!(params.host.as_deref(), Some("localhost:3333"));
    }
}
