using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace MockPool.Stratum;

/// <summary>
/// One miner connection: the handshake for the coin's dialect, the periodic new-block jobs a live
/// pool would push, and the mocked submission verdicts.
/// </summary>
public sealed class StratumSession
{
    private const int Extranonce2Size = 4;
    private const int MaxTrackedShares = 8192;
    private const double MinDifficulty = 1.0;
    private const double MaxDifficulty = 1e15;
    private const double MaxRetargetFactor = 4.0;

    private static readonly object EmptyObject = new();

    private readonly CoinDefinition _coin;
    private readonly MockPoolOptions _options;
    private readonly TcpClient _client;
    private readonly string _peer;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly HashSet<string> _issuedJobs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _acceptedShares = new(StringComparer.Ordinal);
    private readonly Queue<string> _acceptedOrder = new();

    private NetworkStream? _stream;
    private Task? _jobPusher;
    private string? _currentJobId;
    private int _jobCounter;
    private int _height = 100_000;
    private double _difficulty;
    private int _sharesSinceRetarget;

    private bool _authorized;
    private bool _workStarted;
    private string _worker = string.Empty;
    private string _extranonce1 = string.Empty;

    public StratumSession(CoinDefinition coin, MockPoolOptions options, TcpClient client)
    {
        _coin = coin;
        _options = options;
        _client = client;
        _peer = client.Client.RemoteEndPoint?.ToString() ?? "unknown";
    }

    private bool IsPearl => _coin.Protocol == CoinProtocol.Pearl;

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        using var lifetime = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var token = lifetime.Token;

        PoolLog.Debug($"[{_coin.Code}] {_peer} connected");
        try
        {
            _stream = _client.GetStream();
            var reader = new LineReader(_stream);

            while (!token.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync(token).ConfigureAwait(false);
                if (line is null)
                {
                    break;
                }

                if (line.Length == 0)
                {
                    continue;
                }

                await HandleLineAsync(line, token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (IOException exception)
        {
            PoolLog.Debug($"[{_coin.Code}] {_peer} io error: {exception.Message}");
        }
        catch (InvalidDataException exception)
        {
            PoolLog.Warn($"[{_coin.Code}] {_peer} {exception.Message}");
        }
        catch (SocketException exception)
        {
            PoolLog.Debug($"[{_coin.Code}] {_peer} socket error: {exception.Message}");
        }
        catch (ObjectDisposedException)
        {
        }
        finally
        {
            lifetime.Cancel();
            await StopJobPusherAsync().ConfigureAwait(false);
            _client.Dispose();
            PoolLog.Debug($"[{_coin.Code}] {_peer} disconnected");
        }
    }

    private async Task HandleLineAsync(string line, CancellationToken cancellationToken)
    {
        if (!StratumWire.TryParse(line, out var request, out var error))
        {
            if (error is not null)
            {
                PoolLog.Warn($"[{_coin.Code}] {_peer} {error}: {Truncate(line)}");
                await WriteErrorAsync(null, 20, error, cancellationToken).ConfigureAwait(false);
            }

            return;
        }

        if (request!.IsNotification)
        {
            return;
        }

        PoolLog.Debug($"[{_coin.Code}] {_peer} <- {Truncate(line)}");

        await DispatchAsync(request, cancellationToken).ConfigureAwait(false);
    }

    private Task DispatchAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        if (IsPearl)
        {
            return request.Method switch
            {
                "mining.authorize" => OnAuthorizeAsync(request, cancellationToken),
                "mining.submit" => OnSubmitAsync(request, cancellationToken),
                _ => WriteErrorAsync(request.Id, 20, "method not supported", cancellationToken),
            };
        }

        return request.Method switch
        {
            "mining.subscribe" => OnSubscribeAsync(request, cancellationToken),
            "mining.authorize" => OnAuthorizeAsync(request, cancellationToken),
            "mining.submit" => OnSubmitAsync(request, cancellationToken),
            "mining.configure" => ReplyResultAsync(request, EmptyObject, cancellationToken),
            "mining.suggest_difficulty" => ReplyResultAsync(request, true, cancellationToken),
            "mining.extranonce.subscribe" => ReplyResultAsync(request, true, cancellationToken),
            "mining.ping" => ReplyResultAsync(request, "pong", cancellationToken),
            _ => ReplyErrorAsync(request, StratumErrors.UnknownMethod, cancellationToken),
        };
    }

    // --- Classic V1 -------------------------------------------------------------------------------

    private async Task OnSubscribeAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        _extranonce1 = RandomNumberGenerator.GetHexString(8);
        var subscriptionId = RandomNumberGenerator.GetHexString(8);

        var result = new object?[]
        {
            new object?[]
            {
                new object?[] { "mining.set_difficulty", subscriptionId },
                new object?[] { "mining.notify", subscriptionId },
            },
            _extranonce1,
            Extranonce2Size,
        };

        await ReplyResultAsync(request, result, cancellationToken).ConfigureAwait(false);
        PoolLog.Info($"[{_coin.Code}] {_peer} subscribed (agent={ParamAt(request, 0) ?? "unknown"})");

        await StartWorkAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task OnAuthorizeAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        if (IsPearl)
        {
            await OnAuthorizePearlAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        var username = ParamAt(request, 0);
        if (username is null || IsUnauthorizedWorkerName(username))
        {
            PoolLog.Info($"[{_coin.Code}] {_peer} authorize refused for '{username ?? "<empty>"}'");
            await ReplyErrorAsync(request, StratumErrors.Unauthorized, cancellationToken).ConfigureAwait(false);
            return;
        }

        _authorized = true;
        _worker = username;
        await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);
        PoolLog.Info($"[{_coin.Code}] {_peer} authorized worker={username}");

        // A classic client may authorize without subscribing; hand out work either way.
        await StartWorkAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task OnAuthorizePearlAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        // Pearl uses named params, but a miner that probed a classic pool may fall back to the
        // positional form, so accept either.
        var wallet = ObjectString(request, "wallet") ?? ParamAt(request, 0);
        if (wallet is null)
        {
            PoolLog.Info($"[{_coin.Code}] {_peer} authorize refused: wallet is missing");
            await WriteErrorAsync(request.Id, 24, "wallet is missing", cancellationToken).ConfigureAwait(false);
            return;
        }

        if (IsUnauthorizedWorkerName(wallet))
        {
            PoolLog.Info($"[{_coin.Code}] {_peer} authorize refused for '{wallet}'");
            await WriteErrorAsync(request.Id, 25, "invalid wallet", cancellationToken).ConfigureAwait(false);
            return;
        }

        _authorized = true;
        _worker = wallet;

        // Pearl pushes the first job BEFORE the authorize ack (spec §4.1).
        await StartWorkAsync(cancellationToken).ConfigureAwait(false);
        await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);

        var workerLabel = ObjectString(request, "worker");
        PoolLog.Info($"[{_coin.Code}] {_peer} authorized wallet={wallet}"
            + (workerLabel is null ? string.Empty : $" worker={workerLabel}"));
    }

    // --- Submissions ------------------------------------------------------------------------------

    private async Task OnSubmitAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        if (IsPearl)
        {
            await OnSubmitPearlAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        // "Not subscribed" means no work has been handed out yet — which, for a client that
        // authorizes without subscribing, happens at authorize.
        if (!_workStarted)
        {
            await ReplyErrorAsync(request, StratumErrors.NotSubscribed, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (!_authorized)
        {
            await ReplyErrorAsync(request, StratumErrors.Unauthorized, cancellationToken).ConfigureAwait(false);
            return;
        }

        var worker = ParamAt(request, 0);
        var jobId = ParamAt(request, 1);
        var nonce = ParamAt(request, 4);
        if (worker is null || jobId is null || nonce is null)
        {
            await ReplyErrorAsync(request, StratumErrors.InvalidParams, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (!string.Equals(worker, _worker, StringComparison.Ordinal))
        {
            await ReplyVerdictAsync(request, StratumVerdict.Unauthorized, jobId, nonce, cancellationToken)
                .ConfigureAwait(false);
            return;
        }

        var superseded = IsSuperseded(jobId);
        var verdict = ShareVerdicts.Resolve(jobId, nonce, superseded);
        if (verdict == StratumVerdict.Accepted && IsDuplicateShare(jobId, nonce))
        {
            verdict = StratumVerdict.Duplicate;
        }

        await ReplyVerdictAsync(request, verdict, jobId, nonce, cancellationToken).ConfigureAwait(false);
    }

    private async Task OnSubmitPearlAsync(StratumRequest request, CancellationToken cancellationToken)
    {
        if (!_authorized)
        {
            await WriteErrorAsync(request.Id, 27, "unauthorized", cancellationToken).ConfigureAwait(false);
            return;
        }

        var jobId = ObjectString(request, "job_id") ?? ParamAt(request, 1);
        var proof = ObjectString(request, "plain_proof") ?? ParamAt(request, 4);
        if (jobId is null || proof is null)
        {
            await WriteErrorAsync(request.Id, 26, "invalid proof", cancellationToken).ConfigureAwait(false);
            return;
        }

        var superseded = IsSuperseded(jobId);
        var verdict = ShareVerdicts.Resolve(jobId, proof, superseded);
        if (verdict == StratumVerdict.Accepted && IsDuplicateShare(jobId, proof))
        {
            verdict = StratumVerdict.Duplicate;
        }

        await ReplyPearlVerdictAsync(request, verdict, jobId, proof, cancellationToken).ConfigureAwait(false);
    }

    private async Task ReplyVerdictAsync(
        StratumRequest request,
        StratumVerdict verdict,
        string jobId,
        string nonce,
        CancellationToken cancellationToken)
    {
        var detail = $"worker={_worker} job={jobId} nonce={nonce}";

        if (verdict == StratumVerdict.Accepted)
        {
            _sharesSinceRetarget++;
            PoolLog.Info($"[{_coin.Code}] ACCEPTED {detail}");
            await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (verdict == StratumVerdict.Rejected)
        {
            PoolLog.Info($"[{_coin.Code}] REJECTED {detail} reason=rejected");
            await ReplyResultAsync(request, false, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (StratumErrors.TryFor(verdict, out var error))
        {
            PoolLog.Info($"[{_coin.Code}] REJECTED {detail} reason={error.Message} ({error.Code})");
            await ReplyErrorAsync(request, error, cancellationToken).ConfigureAwait(false);
            return;
        }

        await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);
    }

    private async Task ReplyPearlVerdictAsync(
        StratumRequest request,
        StratumVerdict verdict,
        string jobId,
        string proof,
        CancellationToken cancellationToken)
    {
        var detail = $"wallet={_worker} job={jobId} proof={Truncate(proof)}";

        if (verdict == StratumVerdict.Accepted)
        {
            _sharesSinceRetarget++;
            PoolLog.Info($"[{_coin.Code}] ACCEPTED {detail}");
            await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (verdict == StratumVerdict.Rejected)
        {
            PoolLog.Info($"[{_coin.Code}] REJECTED {detail} reason=rejected");
            await ReplyResultAsync(request, false, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (PearlErrors.TryFor(verdict, out var error))
        {
            PoolLog.Info($"[{_coin.Code}] REJECTED {detail} reason={error.Message} ({error.Code})");
            await WriteErrorAsync(request.Id, error.Code, error.Message, cancellationToken).ConfigureAwait(false);
            return;
        }

        await ReplyResultAsync(request, true, cancellationToken).ConfigureAwait(false);
    }

    // --- Work -------------------------------------------------------------------------------------

    /// <summary>
    /// Pushes the first job (and, for classic V1, the difficulty), then keeps fresh blocks coming.
    /// Idempotent per connection, so it does not matter whether work was requested by subscribe or
    /// by authorize.
    /// </summary>
    private async Task StartWorkAsync(CancellationToken cancellationToken)
    {
        if (_workStarted)
        {
            return;
        }

        _workStarted = true;
        _difficulty = IsPearl ? _options.PearlDifficulty : _options.Difficulty;
        var (jobId, parameters) = IssueJob();

        if (!IsPearl)
        {
            await WriteAsync(
                StratumWire.Notification("mining.set_difficulty", new object?[] { _options.Difficulty }),
                cancellationToken).ConfigureAwait(false);
        }

        await WriteAsync(StratumWire.Notification("mining.notify", parameters), cancellationToken).ConfigureAwait(false);
        PoolLog.Info($"[{_coin.Code}] {_peer} work started job={jobId}");

        if (_options.JobInterval > TimeSpan.Zero && _jobPusher is null)
        {
            _jobPusher = PushJobsAsync(cancellationToken);
        }
    }

    /// <summary>
    /// Emits a new job on every tick, the way a live pool hands out the next block. Jobs that were
    /// pushed earlier are superseded, so a late submission against them is reported stale.
    /// </summary>
    private async Task PushJobsAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(_options.JobInterval, cancellationToken).ConfigureAwait(false);

                var changed = Retarget();
                if (!IsPearl && changed)
                {
                    await WriteAsync(
                        StratumWire.Notification("mining.set_difficulty", new object?[] { _difficulty }),
                        cancellationToken).ConfigureAwait(false);
                }

                var (jobId, parameters) = IssueJob();
                await WriteAsync(StratumWire.Notification("mining.notify", parameters), cancellationToken)
                    .ConfigureAwait(false);

                PoolLog.Info($"[{_coin.Code}] {_peer} new job {jobId} (next block)");
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (IOException exception)
        {
            PoolLog.Debug($"[{_coin.Code}] {_peer} job push stopped: {exception.Message}");
        }
        catch (ObjectDisposedException)
        {
        }
    }

    private async Task StopJobPusherAsync()
    {
        if (_jobPusher is null)
        {
            return;
        }

        try
        {
            await _jobPusher.ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
        }

        _jobPusher = null;
    }

    private (string JobId, object Parameters) IssueJob()
    {
        var index = ++_jobCounter;
        string jobId;
        object parameters;

        if (IsPearl)
        {
            jobId = $"{RandomNumberGenerator.GetHexString(8, lowercase: true)}_{index}";
            parameters = new PearlJob(jobId, PearlJob.TargetFor(_difficulty), ++_height).ToNotifyParams();
        }
        else
        {
            jobId = $"{_coin.Ticker}-{index}";
            parameters = new StratumJob(jobId).ToNotifyParams();
        }

        _issuedJobs.Add(jobId);
        _currentJobId = jobId;
        return (jobId, parameters);
    }

    private bool IsSuperseded(string jobId) =>
        _issuedJobs.Contains(jobId) && !string.Equals(jobId, _currentJobId, StringComparison.Ordinal);

    /// <summary>
    /// Vardiff: nudges the share difficulty toward one accepted share per
    /// <see cref="MockPoolOptions.TargetShareInterval"/>. Too many shares raises it (tougher target),
    /// none lowers it. The per-step change is clamped so it converges instead of oscillating.
    /// </summary>
    private bool Retarget()
    {
        var window = _options.JobInterval.TotalSeconds;
        var wanted = _options.TargetShareInterval.TotalSeconds;
        var shares = _sharesSinceRetarget;
        _sharesSinceRetarget = 0;

        if (window <= 0 || wanted <= 0)
        {
            return false;
        }

        var observed = shares / window;                          // shares per second
        var target = 1.0 / wanted;                               // shares per second we want
        var ratio = observed <= 0 ? 1.0 / MaxRetargetFactor : observed / target;
        ratio = Math.Clamp(ratio, 1.0 / MaxRetargetFactor, MaxRetargetFactor);

        var next = Math.Clamp(_difficulty * ratio, MinDifficulty, MaxDifficulty);
        if (Math.Abs(next - _difficulty) <= _difficulty * 0.001)
        {
            return false;
        }

        _difficulty = next;
        PoolLog.Info($"[{_coin.Code}] {_peer} vardiff difficulty={_difficulty:0.##} ({observed:0.##} shares/s)");
        return true;
    }

    /// <summary>
    /// Records an accepted share's (job id, tag) and reports whether it was already submitted. The
    /// raw proof can be hundreds of KB, so only its SHA-256 is kept; the bound keeps a hammering
    /// client from growing the set without limit.
    /// </summary>
    private bool IsDuplicateShare(string jobId, string tag)
    {
        var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(tag)));
        var key = $"{jobId}|{digest}";

        if (!_acceptedShares.Add(key))
        {
            return true;
        }

        _acceptedOrder.Enqueue(key);
        if (_acceptedOrder.Count > MaxTrackedShares)
        {
            _acceptedShares.Remove(_acceptedOrder.Dequeue());
        }

        return false;
    }

    // --- Wire helpers -----------------------------------------------------------------------------

    private Task ReplyResultAsync(StratumRequest request, object? result, CancellationToken cancellationToken) =>
        WriteAsync(StratumWire.Response(request.Id, result), cancellationToken);

    private Task ReplyErrorAsync(StratumRequest request, StratumError error, CancellationToken cancellationToken) =>
        WriteErrorAsync(request.Id, error.Code, error.Message, cancellationToken);

    /// <summary>Writes an error in the coin's dialect envelope (classic array vs Pearl object).</summary>
    private Task WriteErrorAsync(JsonElement? id, int code, string message, CancellationToken cancellationToken) =>
        WriteAsync(
            IsPearl ? StratumWire.PearlError(id, code, message) : StratumWire.Error(id, code, message),
            cancellationToken);

    /// <summary>Serializes writes: the job pusher and the request path share the socket.</summary>
    private async Task WriteAsync(string message, CancellationToken cancellationToken)
    {
        var stream = _stream;
        if (stream is null)
        {
            return;
        }

        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var bytes = Encoding.UTF8.GetBytes(message + "\n");
            await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
            await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private static string? ParamAt(StratumRequest request, int index)
    {
        if (request.Params.ValueKind != JsonValueKind.Array || request.Params.GetArrayLength() <= index)
        {
            return null;
        }

        var element = request.Params[index];
        if (element.ValueKind != JsonValueKind.String)
        {
            return null;
        }

        var value = element.GetString();
        return string.IsNullOrWhiteSpace(value) ? null : value;
    }

    private static string? ObjectString(StratumRequest request, string name)
    {
        if (request.Params.ValueKind != JsonValueKind.Object
            || !request.Params.TryGetProperty(name, out var element)
            || element.ValueKind != JsonValueKind.String)
        {
            return null;
        }

        var value = element.GetString();
        return string.IsNullOrWhiteSpace(value) ? null : value;
    }

    private static bool IsUnauthorizedWorkerName(string username)
    {
        var worker = username;
        var dot = username.LastIndexOf('.');
        if (dot >= 0 && dot < username.Length - 1)
        {
            worker = username[(dot + 1)..];
        }

        return worker.Equals("unauthorized", StringComparison.OrdinalIgnoreCase);
    }

    private static string Truncate(string value) => value.Length <= 200 ? value : value[..200] + "...";
}
