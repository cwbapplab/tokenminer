using Microsoft.Extensions.Logging;
using TokenMiner.Application.Treasury;
using TokenMiner.Application.Treasury.Abstractions;

namespace TokenMiner.Infrastructure.Integrations;

/// <summary>
/// Books a conversion for an operator to settle by hand. Used until a live exchange adapter is
/// configured: the transaction is recorded immediately and completed once the operator has
/// actually moved the funds.
/// </summary>
internal sealed class ManualConversionProvider : IConversionProvider
{
    public string Name => "manual";

    public Task<ConversionQuote?> QuoteAsync(
        ConversionQuoteRequest request,
        CancellationToken cancellationToken) =>
        // An operator supplies the rate, so there is nothing to quote up front.
        Task.FromResult<ConversionQuote?>(null);

    public Task<ConversionResult> ExecuteAsync(
        ConversionExecutionRequest request,
        CancellationToken cancellationToken) =>
        Task.FromResult(ConversionResult.AwaitingManualSettlement(
            "Awaiting manual settlement: complete this conversion once the funds have been moved."));
}

/// <summary>
/// Deterministic in-process conversion at a fixed rate. Exists so the whole pipeline can be
/// exercised without touching a real exchange.
/// </summary>
internal sealed class SimulatedConversionProvider(
    TreasuryOptions options,
    ILogger<SimulatedConversionProvider> logger) : IConversionProvider
{
    public string Name => "simulated";

    public Task<ConversionQuote?> QuoteAsync(
        ConversionQuoteRequest request,
        CancellationToken cancellationToken)
    {
        var rate = options.SimulatedConversionRate;
        var fees = options.SimulatedConversionFeeRate * request.SourceAmount;

        return Task.FromResult<ConversionQuote?>(new ConversionQuote(
            rate,
            decimal.Round(request.SourceAmount * rate, 8, MidpointRounding.ToZero),
            decimal.Round(fees, 8, MidpointRounding.ToZero),
            "Simulated rate."));
    }

    public async Task<ConversionResult> ExecuteAsync(
        ConversionExecutionRequest request,
        CancellationToken cancellationToken)
    {
        logger.LogInformation(
            "Simulating the conversion of {Amount} {Source} into {Destination} (key {Key}).",
            request.SourceAmount,
            request.SourceCoinCode,
            request.DestinationCoinCode,
            request.IdempotencyKey);

        await Task.Yield();

        var quote = await QuoteAsync(
            new ConversionQuoteRequest(
                request.SourceCoinCode,
                request.DestinationCoinCode,
                request.SourceAmount),
            cancellationToken);

        var priced = quote!;

        return ConversionResult.Completed(
            priced.DestinationAmount,
            priced.ExchangeRate,
            priced.Fees,
            destinationTransactionId: $"simulated-{request.TransactionId:N}");
    }
}
