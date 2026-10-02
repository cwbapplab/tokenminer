using System.Net;
using System.Text;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Infrastructure.Integrations;
using Xunit;

namespace TokenMiner.UnitTests.Treasury;

/// <summary>
/// Verifies the Kryptex adapter against the pool's documented response shapes, and that the
/// withdrawal path stays inert unless an operator opts in.
/// </summary>
public sealed class KryptexPoolPayoutProviderTests
{
    private static readonly PoolPayoutContext Context = new(
        Guid.NewGuid(),
        "kryptex-qtc",
        "kryptex",
        "qtc",
        "krxYR9NJVQ",
        "https://pool.kryptex.com");

    private sealed class StubHandler(HttpStatusCode status, string body) : HttpMessageHandler
    {
        public Uri? LastRequest { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            LastRequest = request.RequestUri;

            return Task.FromResult(new HttpResponseMessage(status)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
        }
    }

    private static KryptexPoolPayoutProvider CreateProvider(
        string body,
        HttpStatusCode status = HttpStatusCode.OK,
        IPoolWithdrawalClient? withdrawal = null) =>
        new(
            new HttpClient(new StubHandler(status, body)),
            withdrawal ?? new DisabledPoolWithdrawalClient(),
            NullLogger<KryptexPoolPayoutProvider>.Instance);

    [Fact]
    public async Task ReadsTheBalanceFromTheDocumentedShape()
    {
        var provider = CreateProvider("""{"total":12.5,"threshold":1.5,"unconfirmed":0.5,"confirmed":12}""");

        var balance = await provider.GetBalanceAsync(Context, CancellationToken.None);

        balance.Total.Should().Be(12.5m);
        balance.Confirmed.Should().Be(12m);
        balance.Unconfirmed.Should().Be(0.5m);
        balance.Threshold.Should().Be(1.5m);
    }

    [Fact]
    public async Task ReadsPayoutHistoryAndMarksSettledStatuses()
    {
        const string body = """
            {
              "count": 2,
              "results": [
                { "date": 1767225600, "received": 3.5, "txid": "tx-a", "status": "confirmed" },
                { "date": 1767225600, "received": 1.25, "txid": "tx-b", "status": "pending" }
              ]
            }
            """;

        var provider = CreateProvider(body);

        var payouts = await provider.GetPayoutsAsync(Context, CancellationToken.None);

        payouts.Should().HaveCount(2);

        payouts[0].TransactionHash.Should().Be("tx-a");
        payouts[0].Amount.Should().Be(3.5m);
        payouts[0].IsSettled.Should().BeTrue();
        payouts[0].ReceivedAt.Should().NotBeNull();

        payouts[1].TransactionHash.Should().Be("tx-b");
        payouts[1].IsSettled.Should().BeFalse();
        payouts[1].Confirmations.Should().Be(0);
    }

    [Fact]
    public async Task RequestsTheCoinSlugAndAddressInThePath()
    {
        var handler = new StubHandler(HttpStatusCode.OK, "{}");
        var provider = new KryptexPoolPayoutProvider(
            new HttpClient(handler),
            new DisabledPoolWithdrawalClient(),
            NullLogger<KryptexPoolPayoutProvider>.Instance);

        await provider.GetBalanceAsync(Context, CancellationToken.None);

        handler.LastRequest!.AbsolutePath.Should().Be("/qtc/api/v1/miner/balance/krxYR9NJVQ");
    }

    [Fact]
    public async Task RefusesToReadWithoutAPayoutAddress()
    {
        var provider = CreateProvider("{}");
        var contextWithoutAddress = Context with { PayoutAddress = null };

        var act = async () => await provider.GetBalanceAsync(contextWithoutAddress, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task DoesNotWithdrawWhileBrowserWithdrawalsAreDisabled()
    {
        var provider = CreateProvider("{}");

        var result = await provider.RequestWithdrawalAsync(Context, 5m, CancellationToken.None);

        result.Requested.Should().BeFalse();
    }
}
