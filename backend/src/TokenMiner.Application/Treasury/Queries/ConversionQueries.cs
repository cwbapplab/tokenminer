using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Treasury.Abstractions;
using TokenMiner.Application.Treasury.Models;
using TokenMiner.Domain.Treasury;
using TokenMiner.Domain.Treasury.Enums;

namespace TokenMiner.Application.Treasury.Queries;

// --- Pool payouts ------------------------------------------------------------------------

public sealed record ListPoolPayoutsQuery(int Limit) : IRequest<IReadOnlyList<PoolPayoutDto>>;

internal sealed class ListPoolPayoutsQueryHandler(IPoolPayoutRepository payouts)
    : IRequestHandler<ListPoolPayoutsQuery, IReadOnlyList<PoolPayoutDto>>
{
    public async Task<IReadOnlyList<PoolPayoutDto>> Handle(
        ListPoolPayoutsQuery request,
        CancellationToken cancellationToken) =>
        (await payouts.ListRecentAsync(Math.Clamp(request.Limit, 1, 500), cancellationToken))
            .Select(payout => payout.ToDto())
            .ToList();
}

// --- Providers ---------------------------------------------------------------------------

public sealed record ListConversionProvidersQuery : IRequest<IReadOnlyList<ConversionProviderDto>>;

internal sealed class ListConversionProvidersQueryHandler(IConversionRepository conversions)
    : IRequestHandler<ListConversionProvidersQuery, IReadOnlyList<ConversionProviderDto>>
{
    public async Task<IReadOnlyList<ConversionProviderDto>> Handle(
        ListConversionProvidersQuery request,
        CancellationToken cancellationToken) =>
        (await conversions.ListProvidersAsync(cancellationToken))
            .OrderByDescending(provider => provider.Priority)
            .Select(provider => provider.ToDto())
            .ToList();
}

public sealed record CreateConversionProviderCommand(
    string Name,
    string BaseUrl,
    int Priority,
    string? CredentialRef,
    string? SupportedFeatures) : IRequest<ConversionProviderDto>;

public sealed class CreateConversionProviderCommandValidator : AbstractValidator<CreateConversionProviderCommand>
{
    public CreateConversionProviderCommandValidator()
    {
        RuleFor(command => command.Name).NotEmpty().MaximumLength(64);
        RuleFor(command => command.BaseUrl).NotEmpty().MaximumLength(512);
        RuleFor(command => command.Priority).GreaterThanOrEqualTo(0);
        RuleFor(command => command.CredentialRef).MaximumLength(256);
    }
}

internal sealed class CreateConversionProviderCommandHandler(
    IConversionRepository conversions,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreateConversionProviderCommand, ConversionProviderDto>
{
    public async Task<ConversionProviderDto> Handle(
        CreateConversionProviderCommand request,
        CancellationToken cancellationToken)
    {
        var name = request.Name.Trim().ToLowerInvariant();

        if (await conversions.GetProviderByNameAsync(name, cancellationToken) is not null)
        {
            throw new ConflictException($"A conversion provider named '{name}' already exists.");
        }

        var provider = new ConversionProvider(
            Guid.NewGuid(),
            name,
            request.BaseUrl.Trim(),
            request.Priority,
            request.CredentialRef,
            request.SupportedFeatures,
            timeProvider.GetUtcNow());

        conversions.AddProvider(provider);
        await unitOfWork.SaveChangesAsync(cancellationToken);

        return provider.ToDto();
    }
}

public sealed record UpdateConversionProviderCommand(
    Guid Id,
    string BaseUrl,
    int Priority,
    string Status,
    string? CredentialRef,
    string? SupportedFeatures) : IRequest<ConversionProviderDto>;

public sealed class UpdateConversionProviderCommandValidator : AbstractValidator<UpdateConversionProviderCommand>
{
    public UpdateConversionProviderCommandValidator()
    {
        RuleFor(command => command.Id).NotEmpty();
        RuleFor(command => command.BaseUrl).NotEmpty().MaximumLength(512);
        RuleFor(command => command.Priority).GreaterThanOrEqualTo(0);
        RuleFor(command => command.Status)
            .Must(value => TreasuryStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");
    }
}

internal sealed class UpdateConversionProviderCommandHandler(
    IConversionRepository conversions,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdateConversionProviderCommand, ConversionProviderDto>
{
    public async Task<ConversionProviderDto> Handle(
        UpdateConversionProviderCommand request,
        CancellationToken cancellationToken)
    {
        var provider = await conversions.GetProviderByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Conversion provider not found.");

        provider.Update(
            request.BaseUrl.Trim(),
            request.Priority,
            request.CredentialRef,
            request.SupportedFeatures,
            TreasuryStatusDbValues.FromDbValue(request.Status),
            timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return provider.ToDto();
    }
}

// --- Routes ------------------------------------------------------------------------------

public sealed record ListConversionRoutesQuery : IRequest<IReadOnlyList<ConversionRouteDto>>;

internal sealed class ListConversionRoutesQueryHandler(
    IConversionRepository conversions,
    ICoinRepository coins) : IRequestHandler<ListConversionRoutesQuery, IReadOnlyList<ConversionRouteDto>>
{
    public async Task<IReadOnlyList<ConversionRouteDto>> Handle(
        ListConversionRoutesQuery request,
        CancellationToken cancellationToken)
    {
        var providerNames = (await conversions.ListProvidersAsync(cancellationToken))
            .ToDictionary(provider => provider.Id, provider => provider.Name);

        var coinCodes = (await coins.ListAsync(cancellationToken))
            .ToDictionary(coin => coin.Id, coin => coin.Code);

        return (await conversions.ListRoutesAsync(cancellationToken))
            .OrderByDescending(route => route.Priority)
            .Select(route => route.ToDto(
                providerNames.GetValueOrDefault(route.ConversionProviderId, "(unknown)"),
                coinCodes.GetValueOrDefault(route.SourceCoinId, "(unknown)"),
                coinCodes.GetValueOrDefault(route.DestinationCoinId, "(unknown)")))
            .ToList();
    }
}

public sealed record CreateConversionRouteCommand(
    Guid ConversionProviderId,
    Guid SourceCoinId,
    Guid DestinationCoinId,
    string? SourceNetwork,
    string? DestinationNetwork,
    int Priority,
    decimal MinimumAmount) : IRequest<ConversionRouteDto>;

public sealed class CreateConversionRouteCommandValidator : AbstractValidator<CreateConversionRouteCommand>
{
    public CreateConversionRouteCommandValidator()
    {
        RuleFor(command => command.ConversionProviderId).NotEmpty();
        RuleFor(command => command.SourceCoinId).NotEmpty();
        RuleFor(command => command.DestinationCoinId).NotEmpty();
        RuleFor(command => command.Priority).GreaterThanOrEqualTo(0);
        RuleFor(command => command.MinimumAmount).GreaterThanOrEqualTo(0);
        RuleFor(command => command.DestinationCoinId)
            .NotEqual(command => command.SourceCoinId)
            .WithMessage("A route must convert between two different coins.");
    }
}

internal sealed class CreateConversionRouteCommandHandler(
    IConversionRepository conversions,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreateConversionRouteCommand, ConversionRouteDto>
{
    public async Task<ConversionRouteDto> Handle(
        CreateConversionRouteCommand request,
        CancellationToken cancellationToken)
    {
        if (await conversions.GetProviderByIdAsync(request.ConversionProviderId, cancellationToken) is null)
        {
            throw new NotFoundException("Conversion provider not found.");
        }

        var sourceCoin = await coins.GetByIdAsync(request.SourceCoinId, cancellationToken)
            ?? throw new NotFoundException("Source coin not found.");

        var destinationCoin = await coins.GetByIdAsync(request.DestinationCoinId, cancellationToken)
            ?? throw new NotFoundException("Destination coin not found.");

        var route = new ConversionRoute(
            Guid.NewGuid(),
            request.ConversionProviderId,
            request.SourceCoinId,
            request.DestinationCoinId,
            request.SourceNetwork?.Trim(),
            request.DestinationNetwork?.Trim(),
            request.Priority,
            request.MinimumAmount,
            timeProvider.GetUtcNow());

        conversions.AddRoute(route);
        await unitOfWork.SaveChangesAsync(cancellationToken);

        var provider = await conversions.GetProviderByIdAsync(route.ConversionProviderId, cancellationToken);

        return route.ToDto(provider?.Name ?? "(unknown)", sourceCoin.Code, destinationCoin.Code);
    }
}

public sealed record UpdateConversionRouteCommand(
    Guid Id,
    bool Enabled,
    int Priority,
    decimal MinimumAmount) : IRequest<ConversionRouteDto>;

public sealed class UpdateConversionRouteCommandValidator : AbstractValidator<UpdateConversionRouteCommand>
{
    public UpdateConversionRouteCommandValidator()
    {
        RuleFor(command => command.Id).NotEmpty();
        RuleFor(command => command.Priority).GreaterThanOrEqualTo(0);
        RuleFor(command => command.MinimumAmount).GreaterThanOrEqualTo(0);
    }
}

internal sealed class UpdateConversionRouteCommandHandler(
    IConversionRepository conversions,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdateConversionRouteCommand, ConversionRouteDto>
{
    public async Task<ConversionRouteDto> Handle(
        UpdateConversionRouteCommand request,
        CancellationToken cancellationToken)
    {
        var route = await conversions.GetRouteByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Conversion route not found.");

        route.Update(request.Enabled, request.Priority, request.MinimumAmount, timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);

        var provider = await conversions.GetProviderByIdAsync(route.ConversionProviderId, cancellationToken);
        var sourceCoin = await coins.GetByIdAsync(route.SourceCoinId, cancellationToken);
        var destinationCoin = await coins.GetByIdAsync(route.DestinationCoinId, cancellationToken);

        return route.ToDto(
            provider?.Name ?? "(unknown)",
            sourceCoin?.Code ?? "(unknown)",
            destinationCoin?.Code ?? "(unknown)");
    }
}

// --- Transactions ------------------------------------------------------------------------

public sealed record ListConversionsQuery(int Limit) : IRequest<IReadOnlyList<ConversionTransactionDto>>;

internal sealed class ListConversionsQueryHandler(
    IConversionRepository conversions,
    ICoinRepository coins) : IRequestHandler<ListConversionsQuery, IReadOnlyList<ConversionTransactionDto>>
{
    public async Task<IReadOnlyList<ConversionTransactionDto>> Handle(
        ListConversionsQuery request,
        CancellationToken cancellationToken)
    {
        var providerNames = (await conversions.ListProvidersAsync(cancellationToken))
            .ToDictionary(provider => provider.Id, provider => provider.Name);

        var coinCodes = (await coins.ListAsync(cancellationToken))
            .ToDictionary(coin => coin.Id, coin => coin.Code);

        return (await conversions.ListRecentTransactionsAsync(
                Math.Clamp(request.Limit, 1, 500),
                cancellationToken))
            .Select(transaction => transaction.ToDto(
                providerNames.GetValueOrDefault(transaction.ConversionProviderId, "(unknown)"),
                coinCodes.GetValueOrDefault(transaction.SourceCoinId, "(unknown)"),
                coinCodes.GetValueOrDefault(transaction.DestinationCoinId, "(unknown)")))
            .ToList();
    }
}
