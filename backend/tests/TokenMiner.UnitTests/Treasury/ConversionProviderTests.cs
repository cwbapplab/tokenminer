using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using NSubstitute;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Commands;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;
using TokenMiner.Infrastructure.Integrations;
using Xunit;

namespace TokenMiner.UnitTests.Treasury;

public sealed class ConversionProviderTests
{
    private static readonly Guid TransactionId = Guid.NewGuid();

    private static ConversionExecutionRequest Request(decimal amount = 10m) =>
        new(TransactionId, "payout:abc", "prl", "usdt", amount);

    [Fact]
    public async Task SimulatedProviderConvertsAtTheConfiguredRate()
    {
        var provider = new SimulatedConversionProvider(
            new TreasuryOptions { SimulatedConversionRate = 0.5m, SimulatedConversionFeeRate = 0.01m },
            NullLogger<SimulatedConversionProvider>.Instance);

        var result = await provider.ExecuteAsync(Request(amount: 10m), CancellationToken.None);

        result.Outcome.Should().Be(ConversionOutcome.Completed);
        result.ExchangeRate.Should().Be(0.5m);
        result.DestinationAmount.Should().Be(5m);
        result.Fees.Should().Be(0.1m);
        result.DestinationTransactionId.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public async Task SimulatedProviderIsDeterministicForTheSameRequest()
    {
        var provider = new SimulatedConversionProvider(
            new TreasuryOptions(),
            NullLogger<SimulatedConversionProvider>.Instance);

        var first = await provider.ExecuteAsync(Request(), CancellationToken.None);
        var second = await provider.ExecuteAsync(Request(), CancellationToken.None);

        first.DestinationAmount.Should().Be(second.DestinationAmount);
        first.DestinationTransactionId.Should().Be(second.DestinationTransactionId);
    }

    [Fact]
    public async Task ManualProviderBooksNothingAndAsksToBeSettled()
    {
        var provider = new ManualConversionProvider();

        provider.Name.Should().Be("manual");

        var quote = await provider.QuoteAsync(new ConversionQuoteRequest("prl", "usdt", 10m), CancellationToken.None);
        quote.Should().BeNull();

        var result = await provider.ExecuteAsync(Request(), CancellationToken.None);

        // In flight, not finished: an operator completes it once the funds actually move.
        result.Outcome.Should().Be(ConversionOutcome.Pending);
        result.Error.Should().NotBeNullOrWhiteSpace();
    }
}

public sealed class ConversionRouteSelectorTests
{
    private static readonly Guid SourceCoinId = Guid.NewGuid();
    private static readonly Guid DestinationCoinId = Guid.NewGuid();

    private static ConversionProvider CreateProvider(string name, int priority, TreasuryStatus status)
    {
        var provider = new ConversionProvider(
            Guid.NewGuid(),
            name,
            "https://example",
            priority,
            null,
            null,
            DateTimeOffset.UtcNow);

        if (status == TreasuryStatus.Disabled)
        {
            provider.Update("https://example", priority, null, null, TreasuryStatus.Disabled, DateTimeOffset.UtcNow);
        }

        return provider;
    }

    private static ConversionRoute CreateRoute(
        Guid providerId,
        int priority,
        decimal minimumAmount,
        bool enabled = true)
    {
        var route = new ConversionRoute(
            Guid.NewGuid(),
            providerId,
            SourceCoinId,
            DestinationCoinId,
            null,
            null,
            priority,
            minimumAmount,
            DateTimeOffset.UtcNow);

        if (!enabled)
        {
            route.Update(false, priority, minimumAmount, DateTimeOffset.UtcNow);
        }

        return route;
    }

    private static ConversionRouteSelector CreateSelector(
        IReadOnlyList<ConversionProvider> providers,
        IReadOnlyList<ConversionRoute> routes)
    {
        var repository = Substitute.For<IConversionRepository>();
        repository.ListRoutesAsync(Arg.Any<CancellationToken>()).Returns(routes);

        foreach (var provider in providers)
        {
            repository.GetProviderByIdAsync(provider.Id, Arg.Any<CancellationToken>()).Returns(provider);
        }

        return new ConversionRouteSelector(repository);
    }

    [Fact]
    public async Task PicksTheHighestPriorityEligibleRoute()
    {
        var low = CreateProvider("low", 1, TreasuryStatus.Active);
        var high = CreateProvider("high", 10, TreasuryStatus.Active);

        var selector = CreateSelector(
            [low, high],
            [CreateRoute(low.Id, 1, 0m), CreateRoute(high.Id, 10, 0m)]);

        var selection = await selector.SelectAsync(SourceCoinId, 5m, CancellationToken.None);

        selection.Should().NotBeNull();
        selection!.Value.Provider.Name.Should().Be("high");
    }

    [Fact]
    public async Task SkipsRoutesBelowTheirMinimum()
    {
        var provider = CreateProvider("only", 1, TreasuryStatus.Active);
        var selector = CreateSelector([provider], [CreateRoute(provider.Id, 1, minimumAmount: 100m)]);

        var selection = await selector.SelectAsync(SourceCoinId, 5m, CancellationToken.None);

        selection.Should().BeNull();
    }

    [Fact]
    public async Task SkipsDisabledRoutesAndDisabledProviders()
    {
        var disabledRoute = CreateProvider("disabled-route", 5, TreasuryStatus.Active);
        var disabledProvider = CreateProvider("disabled-provider", 9, TreasuryStatus.Disabled);

        var selector = CreateSelector(
            [disabledRoute, disabledProvider],
            [
                CreateRoute(disabledRoute.Id, 5, 0m, enabled: false),
                CreateRoute(disabledProvider.Id, 9, 0m),
            ]);

        var selection = await selector.SelectAsync(SourceCoinId, 5m, CancellationToken.None);

        selection.Should().BeNull();
    }

    [Fact]
    public async Task FallsBackToALowerPriorityProviderWhenTheBestIsDisabled()
    {
        var disabled = CreateProvider("disabled", 10, TreasuryStatus.Disabled);
        var usable = CreateProvider("usable", 1, TreasuryStatus.Active);

        var selector = CreateSelector(
            [disabled, usable],
            [CreateRoute(disabled.Id, 10, 0m), CreateRoute(usable.Id, 1, 0m)]);

        var selection = await selector.SelectAsync(SourceCoinId, 5m, CancellationToken.None);

        selection!.Value.Provider.Name.Should().Be("usable");
    }
}
