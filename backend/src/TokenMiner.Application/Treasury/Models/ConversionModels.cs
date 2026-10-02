using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Application.Treasury.Models;

public sealed record ConversionProviderDto(
    Guid Id,
    string Name,
    string BaseUrl,
    string Status,
    int Priority,
    string? CredentialRef,
    string? SupportedFeatures,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record ConversionRouteDto(
    Guid Id,
    Guid ConversionProviderId,
    string ConversionProviderName,
    Guid SourceCoinId,
    string SourceCoinCode,
    Guid DestinationCoinId,
    string DestinationCoinCode,
    string? SourceNetwork,
    string? DestinationNetwork,
    bool Enabled,
    int Priority,
    decimal MinimumAmount,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record ConversionTransactionDto(
    Guid Id,
    Guid ConversionProviderId,
    string ConversionProviderName,
    string SourceCoinCode,
    decimal SourceAmount,
    string DestinationCoinCode,
    decimal? DestinationAmount,
    decimal? ExchangeRate,
    decimal? Fees,
    string? SourceTransactionId,
    string? DestinationTransactionId,
    string Status,
    string IdempotencyKey,
    string? Error,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    DateTimeOffset? CompletedAt);

public static class ConversionMappings
{
    public static ConversionProviderDto ToDto(this ConversionProvider provider) => new(
        provider.Id,
        provider.Name,
        provider.BaseUrl,
        provider.Status.ToDbValue(),
        provider.Priority,
        provider.CredentialRef,
        provider.SupportedFeatures,
        provider.CreatedAt,
        provider.UpdatedAt);

    public static ConversionRouteDto ToDto(
        this ConversionRoute route,
        string providerName,
        string sourceCoinCode,
        string destinationCoinCode) => new(
        route.Id,
        route.ConversionProviderId,
        providerName,
        route.SourceCoinId,
        sourceCoinCode,
        route.DestinationCoinId,
        destinationCoinCode,
        route.SourceNetwork,
        route.DestinationNetwork,
        route.Enabled,
        route.Priority,
        route.MinimumAmount,
        route.CreatedAt,
        route.UpdatedAt);

    public static ConversionTransactionDto ToDto(
        this ConversionTransaction transaction,
        string providerName,
        string sourceCoinCode,
        string destinationCoinCode) => new(
        transaction.Id,
        transaction.ConversionProviderId,
        providerName,
        sourceCoinCode,
        transaction.SourceAmount,
        destinationCoinCode,
        transaction.DestinationAmount,
        transaction.ExchangeRate,
        transaction.Fees,
        transaction.SourceTransactionId,
        transaction.DestinationTransactionId,
        transaction.Status.ToDbValue(),
        transaction.IdempotencyKey,
        transaction.Error,
        transaction.CreatedAt,
        transaction.UpdatedAt,
        transaction.CompletedAt);
}
