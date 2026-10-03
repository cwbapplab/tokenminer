using System.Text.Json;
using MockPool.Stratum;
using Xunit;

namespace MockPool.Tests;

public sealed class StratumWireTests
{
    [Fact]
    public void Parses_a_request_with_an_id()
    {
        Assert.True(StratumWire.TryParse(
            """{"id":7,"method":"mining.submit","params":["w","j"]}""",
            out var request,
            out var error));

        Assert.Null(error);
        Assert.NotNull(request);
        Assert.False(request!.IsNotification);
        Assert.Equal("mining.submit", request.Method);
        Assert.Equal(7, request.Id.GetInt64());
        Assert.Equal(2, request.Params.GetArrayLength());
    }

    [Fact]
    public void Parses_a_notification()
    {
        Assert.True(StratumWire.TryParse(
            """{"id":null,"method":"mining.notify","params":[1]}""",
            out var request,
            out _));

        Assert.True(request!.IsNotification);
    }

    [Fact]
    public void Reports_malformed_json()
    {
        Assert.False(StratumWire.TryParse("{not json", out var request, out var error));
        Assert.Null(request);
        Assert.NotNull(error);
    }

    [Fact]
    public void Ignores_traffic_that_is_not_a_request()
    {
        Assert.False(StratumWire.TryParse(
            """{"id":1,"result":true,"error":null}""",
            out var request,
            out var error));

        Assert.Null(request);
        Assert.Null(error);
    }

    [Fact]
    public void Serializes_an_error_envelope() =>
        Assert.Equal(
            """{"id":null,"result":null,"error":[23,"Low difficulty share",null]}""",
            StratumWire.Error(null, 23, "Low difficulty share"));

    [Fact]
    public void Serializes_an_accept_envelope()
    {
        using var document = JsonDocument.Parse(StratumWire.Response(null, true));

        Assert.True(document.RootElement.GetProperty("result").GetBoolean());
        Assert.Equal(JsonValueKind.Null, document.RootElement.GetProperty("error").ValueKind);
    }

    [Fact]
    public void Serializes_a_notification_with_a_null_id()
    {
        using var document = JsonDocument.Parse(StratumWire.Notification("mining.set_difficulty", new object?[] { 2.0 }));

        Assert.Equal(JsonValueKind.Null, document.RootElement.GetProperty("id").ValueKind);
        Assert.Equal("mining.set_difficulty", document.RootElement.GetProperty("method").GetString());
        Assert.Equal(2.0, document.RootElement.GetProperty("params")[0].GetDouble());
    }
}
