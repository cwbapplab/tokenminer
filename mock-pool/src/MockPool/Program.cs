using System.Globalization;
using System.Net.Sockets;
using MockPool;
using MockPool.Stratum;

var options = MockPoolOptions.FromEnvironment();
PoolLog.SetLevel(options.Verbosity);

using var shutdown = new CancellationTokenSource();
Console.CancelKeyPress += (_, eventArgs) =>
{
    eventArgs.Cancel = true;
    PoolLog.Info("shutdown requested");
    shutdown.Cancel();
};

StratumServer[] servers =
[
    new(Coins.Pearl, options, options.PearlEndPoint),
    new(Coins.Quantus, options, options.QuantusEndPoint),
];

try
{
    foreach (var server in servers)
    {
        server.Start();
    }

    PoolLog.Info($"mock pool ready (difficulty={options.Difficulty.ToString("0.################", CultureInfo.InvariantCulture)})");
    await Task.WhenAll(servers.Select(server => server.RunAsync(shutdown.Token)));
    return 0;
}
catch (SocketException exception)
{
    PoolLog.Error($"could not bind a listener: {exception.Message}");
    return 1;
}
catch (OperationCanceledException)
{
    return 0;
}
finally
{
    PoolLog.Info("mock pool stopped");
}
