using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Text;
using System.Text.Json.Nodes;
using MockPool.Stratum;
using Xunit;

namespace MockPool.Tests;

public sealed class StratumSessionTests
{
    private const string Worker = "wallet.rig-1";

    // --- Classic V1 (Quantus) ---------------------------------------------------------------------

    [Theory]
    [InlineData("00aabbcc", true, null, null)]
    [InlineData("aabbccdd", true, null, null)]
    [InlineData("ffaabbcc", false, null, null)]
    [InlineData("20aabbcc", null, 20, "Other/unknown")]
    [InlineData("21aabbcc", null, 21, "Job not found")]
    [InlineData("22aabbcc", null, 22, "Duplicate share")]
    [InlineData("23aabbcc", null, 23, "Low difficulty share")]
    [InlineData("24aabbcc", null, 24, "Unauthorized worker")]
    [InlineData("25aabbcc", null, 25, "Not subscribed")]
    public async Task Submit_reports_the_mocked_verdict(string nonce, bool? accepted, int? errorCode, string? message)
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await HandshakeAsync(client);

        var response = await SubmitAsync(client, 3, "QTC-1", nonce);

        if (accepted is not null)
        {
            Assert.Equal(accepted.Value, response["result"]!.GetValue<bool>());
            Assert.Null(response["error"]);
        }
        else
        {
            Assert.Null(response["result"]);
            Assert.Equal(errorCode, response["error"]![0]!.GetValue<int>());
            Assert.Equal(message, response["error"]![1]!.GetValue<string>());
        }
    }

    [Fact]
    public async Task Classic_duplicate_nonce_is_refused()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await HandshakeAsync(client);

        var first = await SubmitAsync(client, 3, "QTC-1", "00aabbcc");
        Assert.True(first["result"]!.GetValue<bool>());

        var second = await SubmitAsync(client, 4, "QTC-1", "00aabbcc");
        Assert.Equal(22, second["error"]![0]!.GetValue<int>());
        Assert.Equal("Duplicate share", second["error"]![1]!.GetValue<string>());
    }

    [Fact]
    public async Task Submit_for_an_unknown_job_is_stale()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await HandshakeAsync(client);

        var response = await SubmitAsync(client, 3, "stale-job-9", "aabbccdd");

        Assert.Equal(21, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Submit_for_another_worker_is_refused()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await HandshakeAsync(client);

        var response = await SubmitAsync(client, 3, "QTC-1", "00aabbcc", worker: "wallet.someone-else");

        Assert.Equal(24, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Submit_before_subscribe_is_refused_with_25()
    {
        await using var client = await ConnectAsync(Coins.Quantus);

        var response = await SubmitAsync(client, 1, "QTC-1", "00aabbcc");

        Assert.Equal(25, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Submit_after_subscribe_but_before_authorize_is_refused_with_24()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await SubscribeAsync(client, 1);

        var response = await SubmitAsync(client, 2, "QTC-1", "00aabbcc");

        Assert.Equal(24, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Authorize_with_the_unauthorized_marker_is_refused()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await client.SendAsync("""{"id":1,"method":"mining.authorize","params":["wallet.unauthorized","x"]}""");

        var response = await client.ReadAsync();

        Assert.Equal(24, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Classic_listener_refuses_object_params()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await client.SendAsync("""{"id":1,"method":"mining.authorize","params":{"wallet":"x"}}""");

        var response = await client.ReadAsync();

        Assert.Equal(24, response["error"]![0]!.GetValue<int>());
    }

    [Fact]
    public async Task Malformed_request_is_reported_and_the_connection_survives()
    {
        await using var client = await ConnectAsync(Coins.Quantus);
        await client.SendAsync("{not json");

        var error = await client.ReadAsync();
        Assert.Equal(20, error["error"]![0]!.GetValue<int>());
        Assert.Equal("Parse error", error["error"]![1]!.GetValue<string>());

        var pong = await client.SendAndReadAsync("""{"id":9,"method":"mining.ping"}""");
        Assert.Equal("pong", pong["result"]!.GetValue<string>());
    }

    [Fact]
    public async Task Classic_notify_carries_a_coin_specific_job_id()
    {
        await using var client = await ConnectAsync(Coins.Quantus);

        var (_, notification) = await SubscribeAsync(client, 1);

        Assert.Equal("mining.notify", notification["method"]!.GetValue<string>());
        Assert.StartsWith("QTC-", notification["params"]![0]!.GetValue<string>());
    }

    [Fact]
    public async Task Classic_authorize_without_subscribe_starts_work_and_accepts_submissions()
    {
        await using var client = await ConnectAsync(Coins.Quantus);

        await client.SendAsync("""{"id":1,"method":"mining.authorize","params":["wallet.rig-1","x"]}""");
        var authorize = await client.ReadAsync();
        Assert.True(authorize["result"]!.GetValue<bool>());

        var difficulty = await client.ReadAsync();
        Assert.Equal("mining.set_difficulty", difficulty["method"]!.GetValue<string>());

        var notify = await client.ReadAsync();
        var jobId = notify["params"]![0]!.GetValue<string>();

        var submit = await SubmitAsync(client, 2, jobId, "00aabbcc");
        Assert.True(submit["result"]!.GetValue<bool>());
    }

    [Fact]
    public async Task The_pool_pushes_new_jobs_and_superseded_jobs_become_stale()
    {
        await using var client = await ConnectAsync(Coins.Quantus, TimeSpan.FromSeconds(2));
        await HandshakeAsync(client);

        var newJobId = await ReadNotifyJobIdAsync(client);
        Assert.NotEqual("QTC-1", newJobId);

        var stale = await SubmitAsync(client, 3, "QTC-1", "aabbccdd");
        Assert.Equal(21, stale["error"]![0]!.GetValue<int>());

        var accepted = await SubmitAsync(client, 4, newJobId, "00aabbcc");
        Assert.True(accepted["result"]!.GetValue<bool>());
    }

    // --- Pearl dialect ----------------------------------------------------------------------------

    [Fact]
    public async Task Pearl_authorize_pushes_a_job_before_the_ack()
    {
        await using var client = await ConnectAsync(Coins.Pearl);

        var jobId = await PearlHandshakeAsync(client);

        Assert.Matches("^[0-9a-f]{8}_[0-9]+$", jobId);
    }

    [Fact]
    public async Task Pearl_notify_carries_a_76_byte_header_and_a_target()
    {
        await using var client = await ConnectAsync(Coins.Pearl);
        await client.SendAsync("""{"id":1,"method":"mining.authorize","params":{"wallet":"prl1pwallet"}}""");

        var notify = await client.ReadAsync();
        var parameters = notify["params"]!;

        Assert.Equal("mining.notify", notify["method"]!.GetValue<string>());
        Assert.Equal(152, parameters["header"]!.GetValue<string>().Length);
        Assert.Equal(64, parameters["target"]!.GetValue<string>().Length);
        Assert.NotNull(parameters["height"]);
    }

    [Fact]
    public async Task Pearl_submit_accepts_a_plain_proof()
    {
        await using var client = await ConnectAsync(Coins.Pearl);
        var jobId = await PearlHandshakeAsync(client);

        var response = await PearlSubmitAsync(client, 2, jobId, "AAAAproof");

        Assert.True(response["result"]!.GetValue<bool>());
        Assert.Null(response["error"]);
    }

    [Fact]
    public async Task Pearl_duplicate_proof_is_refused()
    {
        await using var client = await ConnectAsync(Coins.Pearl);
        var jobId = await PearlHandshakeAsync(client);

        var first = await PearlSubmitAsync(client, 2, jobId, "AAAAproof");
        Assert.True(first["result"]!.GetValue<bool>());

        var second = await PearlSubmitAsync(client, 3, jobId, "AAAAproof");
        Assert.Null(second["result"]);
        Assert.Equal(22, second["error"]!["code"]!.GetValue<int>());
        Assert.Equal("duplicate share", second["error"]!["msg"]!.GetValue<string>());
    }

    [Theory]
    [InlineData("21aabbcc", 21, "stale job")]
    [InlineData("22aabbcc", 22, "duplicate share")]
    [InlineData("23aabbcc", 23, "low difficulty share")]
    [InlineData("26aabbcc", 26, "invalid proof")]
    public async Task Pearl_submit_reports_mocked_object_errors(string proof, int code, string message)
    {
        await using var client = await ConnectAsync(Coins.Pearl);
        var jobId = await PearlHandshakeAsync(client);

        var response = await PearlSubmitAsync(client, 2, jobId, proof);

        Assert.Null(response["result"]);
        Assert.Equal(code, response["error"]!["code"]!.GetValue<int>());
        Assert.Equal(message, response["error"]!["msg"]!.GetValue<string>());
    }

    [Fact]
    public async Task Pearl_accepts_a_large_plain_proof()
    {
        await using var client = await ConnectAsync(Coins.Pearl);
        var jobId = await PearlHandshakeAsync(client);

        // Pearl proofs run to hundreds of KB — well past a small line cap.
        var proof = new string('A', 300_000);
        var response = await PearlSubmitAsync(client, 2, jobId, proof);

        Assert.True(response["result"]!.GetValue<bool>());
    }

    [Fact]
    public async Task Pearl_vardiff_raises_difficulty_when_shares_flood_in()
    {
        await using var client = await ConnectAsync(Coins.Pearl, TimeSpan.FromSeconds(2));
        await client.SendAsync("""{"id":1,"method":"mining.authorize","params":{"wallet":"prl1pwallet"}}""");

        var firstJob = await client.ReadAsync();
        await client.ReadAsync();
        var jobId = firstJob["params"]!["job_id"]!.GetValue<string>();
        var before = TargetOf(firstJob);

        for (var i = 0; i < 4; i++)
        {
            var response = await PearlSubmitAsync(client, 100 + i, jobId, "AAAAproof" + i);
            Assert.True(response["result"]!.GetValue<bool>());
        }

        var next = await ReadPearlNotifyAsync(client);

        Assert.True(TargetOf(next) < before, $"vardiff should make the target harder ({before} -> {TargetOf(next)})");
    }

    private static BigInteger TargetOf(JsonNode notify) =>
        BigInteger.Parse("0" + notify["params"]!["target"]!.GetValue<string>(), NumberStyles.HexNumber);

    [Fact]
    public async Task Pearl_superseded_job_is_stale()
    {
        await using var client = await ConnectAsync(Coins.Pearl, TimeSpan.FromSeconds(2));
        var first = await PearlHandshakeAsync(client);

        var second = await ReadPearlJobIdAsync(client);
        Assert.NotEqual(first, second);

        var stale = await PearlSubmitAsync(client, 2, first, "AAAAproof");
        Assert.Equal(21, stale["error"]!["code"]!.GetValue<int>());
    }

    [Fact]
    public async Task Pearl_refuses_subscribe_and_authorize_without_a_wallet()
    {
        await using var client = await ConnectAsync(Coins.Pearl);

        var subscribe = await client.SendAndReadAsync("""{"id":1,"method":"mining.subscribe","params":[]}""");
        Assert.Equal(20, subscribe["error"]!["code"]!.GetValue<int>());
        Assert.Equal("method not supported", subscribe["error"]!["msg"]!.GetValue<string>());

        var authorize = await client.SendAndReadAsync("""{"id":2,"method":"mining.authorize","params":{"worker":"rig"}}""");
        Assert.Equal(24, authorize["error"]!["code"]!.GetValue<int>());
        Assert.Equal("wallet is missing", authorize["error"]!["msg"]!.GetValue<string>());
    }

    // --- Helpers ----------------------------------------------------------------------------------

    private static async Task<string> PearlHandshakeAsync(TestClient client, string wallet = "prl1pwallet")
    {
        await client.SendAsync(
            $$$"""{"id":1,"method":"mining.authorize","params":{"wallet":"{{{wallet}}}","worker":"rig1","pass":"x","agent":"test/1.0"}}""");

        // Pearl pushes the first job BEFORE the authorize ack.
        var notify = await client.ReadAsync();
        Assert.Equal("mining.notify", notify["method"]!.GetValue<string>());

        var ack = await client.ReadAsync();
        Assert.True(ack["result"]!.GetValue<bool>());

        return notify["params"]!["job_id"]!.GetValue<string>();
    }

    private static Task<JsonNode> PearlSubmitAsync(TestClient client, long id, string jobId, string proof) =>
        client.SendAndReadAsync(
            $$$"""{"id":{{{id}}},"method":"mining.submit","params":{"job_id":"{{{jobId}}}","plain_proof":"{{{proof}}}"}}""");

    private static async Task<JsonNode> ReadPearlNotifyAsync(TestClient client)
    {
        while (true)
        {
            var line = await client.ReadAsync();
            if (line["method"]?.GetValue<string>() == "mining.notify")
            {
                return line;
            }
        }
    }

    private static async Task<string> ReadPearlJobIdAsync(TestClient client)
    {
        while (true)
        {
            var line = await client.ReadAsync();
            if (line["method"]?.GetValue<string>() == "mining.notify")
            {
                return line["params"]!["job_id"]!.GetValue<string>();
            }
        }
    }

    private static async Task<string> ReadNotifyJobIdAsync(TestClient client)
    {
        while (true)
        {
            var line = await client.ReadAsync();
            if (line["method"]?.GetValue<string>() == "mining.notify")
            {
                return line["params"]![0]!.GetValue<string>();
            }
        }
    }

    private static async Task<(JsonNode Response, JsonNode Notification)> SubscribeAsync(TestClient client, long id)
    {
        await client.SendAsync($$"""{"id":{{id}},"method":"mining.subscribe","params":["mock-miner/1.0"]}""");

        var response = await client.ReadAsync();
        Assert.Equal(id, response["id"]!.GetValue<long>());
        Assert.NotNull(response["result"]);

        var difficulty = await client.ReadAsync();
        Assert.Equal("mining.set_difficulty", difficulty["method"]!.GetValue<string>());

        var notification = await client.ReadAsync();
        return (response, notification);
    }

    private static async Task HandshakeAsync(TestClient client)
    {
        await SubscribeAsync(client, 1);

        await client.SendAsync($$"""{"id":2,"method":"mining.authorize","params":["{{Worker}}","x"]}""");
        var authorize = await client.ReadAsync();
        Assert.True(authorize["result"]!.GetValue<bool>());
    }

    private static Task<JsonNode> SubmitAsync(TestClient client, long id, string jobId, string nonce, string worker = Worker) =>
        client.SendAndReadAsync(
            $$"""{"id":{{id}},"method":"mining.submit","params":["{{worker}}","{{jobId}}","deadbeef","65a1b2c3","{{nonce}}"]}""");

    private static async Task<TestClient> ConnectAsync(CoinDefinition coin, TimeSpan? jobInterval = null)
    {
        var options = new MockPoolOptions
        {
            PearlEndPoint = new IPEndPoint(IPAddress.Loopback, 0),
            QuantusEndPoint = new IPEndPoint(IPAddress.Loopback, 0),
            JobInterval = jobInterval ?? TimeSpan.FromSeconds(30),
        };

        var endpoint = coin.Code == "prl" ? options.PearlEndPoint : options.QuantusEndPoint;
        var server = new StratumServer(coin, options, endpoint);
        server.Start();

        var lifetime = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var accept = server.RunAsync(lifetime.Token);

        var tcp = new TcpClient();
        await tcp.ConnectAsync(IPAddress.Loopback, server.EndPoint.Port, lifetime.Token);
        return new TestClient(tcp, accept, lifetime);
    }

    private sealed class TestClient : IAsyncDisposable
    {
        private readonly TcpClient _tcp;
        private readonly NetworkStream _stream;
        private readonly StreamReader _reader;
        private readonly Task _accept;
        private readonly CancellationTokenSource _lifetime;

        public TestClient(TcpClient tcp, Task accept, CancellationTokenSource lifetime)
        {
            _tcp = tcp;
            _accept = accept;
            _lifetime = lifetime;
            _stream = tcp.GetStream();
            _reader = new StreamReader(_stream, Encoding.UTF8);
        }

        public async Task SendAsync(string json)
        {
            var bytes = Encoding.UTF8.GetBytes(json + "\n");
            await _stream.WriteAsync(bytes, _lifetime.Token);
            await _stream.FlushAsync(_lifetime.Token);
        }

        public async Task<JsonNode> SendAndReadAsync(string json)
        {
            await SendAsync(json);
            return await ReadAsync();
        }

        public async Task<JsonNode> ReadAsync()
        {
            var line = await _reader.ReadLineAsync(_lifetime.Token);
            if (line is null)
            {
                throw new InvalidOperationException("The pool closed the connection.");
            }

            return JsonNode.Parse(line) ?? throw new InvalidOperationException("Empty response.");
        }

        public async ValueTask DisposeAsync()
        {
            _tcp.Dispose();
            _lifetime.Cancel();
            await _accept;
            _lifetime.Dispose();
        }
    }
}
