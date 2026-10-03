using System.Net;
using System.Net.Sockets;

namespace MockPool.Stratum;

/// <summary>Accepts miner connections for one coin and runs a session per connection.</summary>
public sealed class StratumServer
{
    private readonly CoinDefinition _coin;
    private readonly MockPoolOptions _options;
    private readonly TcpListener _listener;

    public StratumServer(CoinDefinition coin, MockPoolOptions options, IPEndPoint endPoint)
    {
        _coin = coin;
        _options = options;
        _listener = new TcpListener(endPoint);
    }

    public CoinDefinition Coin => _coin;

    /// <summary>The bound endpoint. Its port is resolved only after <see cref="Start"/>.</summary>
    public IPEndPoint EndPoint => (IPEndPoint)_listener.LocalEndpoint;

    public void Start()
    {
        _listener.Start();
        PoolLog.Info($"[{_coin.Code}] {_coin.Name} listening on {EndPoint}");
    }

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        var sessions = new List<Task>();
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var client = await _listener.AcceptTcpClientAsync(cancellationToken).ConfigureAwait(false);
                client.NoDelay = true;
                sessions.Add(RunSessionAsync(client, cancellationToken));
                sessions.RemoveAll(static session => session.IsCompleted);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (SocketException exception) when (cancellationToken.IsCancellationRequested)
        {
            PoolLog.Debug($"[{_coin.Code}] listener stopped: {exception.Message}");
        }
        catch (ObjectDisposedException)
        {
        }
        finally
        {
            _listener.Stop();
        }

        await Task.WhenAll(sessions).ConfigureAwait(false);
    }

    private async Task RunSessionAsync(TcpClient client, CancellationToken cancellationToken)
    {
        try
        {
            await new StratumSession(_coin, _options, client).RunAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            PoolLog.Error($"[{_coin.Code}] session failed: {exception.Message}");
            client.Dispose();
        }
    }
}
