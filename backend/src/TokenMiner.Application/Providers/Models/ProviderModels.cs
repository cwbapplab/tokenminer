using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Providers.Abstractions;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Providers;
using TokenMiner.Domain.Providers.Enums;

namespace TokenMiner.Application.Providers.Models;

public sealed record LlmProviderDepositAccountDto(
    Guid Id,
    Guid CoinId,
    string CoinCode,
    string Network,
    string DepositAddress,
    string Status);

public sealed record LlmProviderDto(
    Guid Id,
    string Name,
    string Endpoint,
    string? CredentialRef,
    string Status,
    decimal? Balance,
    decimal? ReservedBalance,
    decimal? TotalBalance,
    DateTimeOffset? BalanceCheckedAt,
    IReadOnlyList<LlmProviderDepositAccountDto> DepositAccounts,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record LlmProviderModelDto(
    Guid Id,
    string ModelId,
    string? Name,
    decimal? InputCost,
    decimal? OutputCost,
    decimal? CachedInputCost,
    string Currency,
    int? ContextLength,
    string? Capabilities,
    string Status,
    DateTimeOffset LastSyncedAt);

public sealed record ProviderDepositDto(
    Guid Id,
    Guid LlmProviderId,
    Guid CoinId,
    string Network,
    string Address,
    decimal Amount,
    string? TransactionHash,
    int Confirmations,
    string Status,
    decimal? ProviderCreditBefore,
    decimal? ProviderCreditAfter,
    string IdempotencyKey,
    string? Error,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    DateTimeOffset? ConfirmedAt);

public sealed record ProviderBalanceSnapshotDto(
    Guid Id,
    Guid LlmProviderId,
    decimal Balance,
    decimal ReservedBalance,
    decimal TotalBalance,
    DateTimeOffset CheckedAt);

public static class ProviderMappings
{
    public static ProviderDepositDto ToDto(this ProviderDeposit deposit) => new(
        deposit.Id,
        deposit.LlmProviderId,
        deposit.CoinId,
        deposit.Network,
        deposit.Address,
        deposit.Amount,
        deposit.TransactionHash,
        deposit.Confirmations,
        deposit.Status.ToDbValue(),
        deposit.ProviderCreditBefore,
        deposit.ProviderCreditAfter,
        deposit.IdempotencyKey,
        deposit.Error,
        deposit.CreatedAt,
        deposit.UpdatedAt,
        deposit.ConfirmedAt);
}

// --- Queries -----------------------------------------------------------------------------

public sealed record ListLlmProvidersQuery : IRequest<IReadOnlyList<LlmProviderDto>>;

internal sealed class ListLlmProvidersQueryHandler(
    ILlmProviderRepository providers,
    ICoinRepository coins) : IRequestHandler<ListLlmProvidersQuery, IReadOnlyList<LlmProviderDto>>
{
    public async Task<IReadOnlyList<LlmProviderDto>> Handle(
        ListLlmProvidersQuery request,
        CancellationToken cancellationToken)
    {
        var coinCodes = (await coins.ListAsync(cancellationToken)).ToDictionary(coin => coin.Id, coin => coin.Code);

        var result = new List<LlmProviderDto>();

        foreach (var provider in await providers.ListProvidersAsync(cancellationToken))
        {
            var accounts = (await providers.ListDepositAccountsAsync(cancellationToken))
                .Where(account => account.LlmProviderId == provider.Id)
                .Select(account => new LlmProviderDepositAccountDto(
                    account.Id,
                    account.CoinId,
                    coinCodes.GetValueOrDefault(account.CoinId, "(unknown)"),
                    account.Network,
                    account.DepositAddress,
                    account.Status.ToDbValue()))
                .ToList();

            var balance = await providers.GetLatestBalanceAsync(provider.Id, cancellationToken);

            result.Add(new LlmProviderDto(
                provider.Id,
                provider.Name,
                provider.Endpoint,
                provider.CredentialRef,
                provider.Status.ToDbValue(),
                balance?.Balance,
                balance?.ReservedBalance,
                balance?.TotalBalance,
                balance?.CheckedAt,
                accounts,
                provider.CreatedAt,
                provider.UpdatedAt));
        }

        return result;
    }
}

public sealed record ListProviderModelsQuery(Guid LlmProviderId) : IRequest<IReadOnlyList<LlmProviderModelDto>>;

internal sealed class ListProviderModelsQueryHandler(ILlmProviderRepository providers)
    : IRequestHandler<ListProviderModelsQuery, IReadOnlyList<LlmProviderModelDto>>
{
    public async Task<IReadOnlyList<LlmProviderModelDto>> Handle(
        ListProviderModelsQuery request,
        CancellationToken cancellationToken) =>
        (await providers.ListModelsAsync(request.LlmProviderId, cancellationToken))
            .OrderBy(model => model.ModelId, StringComparer.Ordinal)
            .Select(model => new LlmProviderModelDto(
                model.Id,
                model.ModelId,
                model.Name,
                model.InputCost,
                model.OutputCost,
                model.CachedInputCost,
                model.Currency,
                model.ContextLength,
                model.Capabilities,
                model.Status.ToDbValue(),
                model.LastSyncedAt))
            .ToList();
}

public sealed record ListProviderDepositsQuery(int Limit) : IRequest<IReadOnlyList<ProviderDepositDto>>;

internal sealed class ListProviderDepositsQueryHandler(IProviderDepositRepository deposits)
    : IRequestHandler<ListProviderDepositsQuery, IReadOnlyList<ProviderDepositDto>>
{
    public async Task<IReadOnlyList<ProviderDepositDto>> Handle(
        ListProviderDepositsQuery request,
        CancellationToken cancellationToken) =>
        (await deposits.ListRecentAsync(Math.Clamp(request.Limit, 1, 500), cancellationToken))
            .Select(deposit => deposit.ToDto())
            .ToList();
}

// --- Configuration -----------------------------------------------------------------------

public sealed record CreateLlmProviderCommand(string Name, string Endpoint, string? CredentialRef)
    : IRequest<LlmProviderDto>;

public sealed class CreateLlmProviderCommandValidator : AbstractValidator<CreateLlmProviderCommand>
{
    public CreateLlmProviderCommandValidator()
    {
        RuleFor(command => command.Name).NotEmpty().MaximumLength(64);
        RuleFor(command => command.Endpoint).NotEmpty().MaximumLength(512);
        RuleFor(command => command.CredentialRef).MaximumLength(256);
    }
}

internal sealed class CreateLlmProviderCommandHandler(
    ILlmProviderRepository providers,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreateLlmProviderCommand, LlmProviderDto>
{
    public async Task<LlmProviderDto> Handle(
        CreateLlmProviderCommand request,
        CancellationToken cancellationToken)
    {
        var name = request.Name.Trim().ToLowerInvariant();

        if (await providers.GetProviderByNameAsync(name, cancellationToken) is not null)
        {
            throw new ConflictException($"An LLM provider named '{name}' already exists.");
        }

        var provider = new LlmProvider(
            Guid.NewGuid(),
            name,
            request.Endpoint.Trim(),
            request.CredentialRef,
            timeProvider.GetUtcNow());

        providers.AddProvider(provider);
        await unitOfWork.SaveChangesAsync(cancellationToken);

        return new LlmProviderDto(
            provider.Id,
            provider.Name,
            provider.Endpoint,
            provider.CredentialRef,
            provider.Status.ToDbValue(),
            null,
            null,
            null,
            null,
            [],
            provider.CreatedAt,
            provider.UpdatedAt);
    }
}

public sealed record UpdateLlmProviderCommand(
    Guid Id,
    string Name,
    string Endpoint,
    string? CredentialRef,
    string Status) : IRequest<LlmProviderDto>;

public sealed class UpdateLlmProviderCommandValidator : AbstractValidator<UpdateLlmProviderCommand>
{
    public UpdateLlmProviderCommandValidator()
    {
        RuleFor(command => command.Id).NotEmpty();
        RuleFor(command => command.Name).NotEmpty().MaximumLength(64);
        RuleFor(command => command.Endpoint).NotEmpty().MaximumLength(512);
        RuleFor(command => command.Status)
            .Must(value => MiningStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");
    }
}

internal sealed class UpdateLlmProviderCommandHandler(
    ILlmProviderRepository providers,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdateLlmProviderCommand, LlmProviderDto>
{
    public async Task<LlmProviderDto> Handle(
        UpdateLlmProviderCommand request,
        CancellationToken cancellationToken)
    {
        var provider = await providers.GetProviderByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("LLM provider not found.");

        provider.Update(
            request.Name.Trim().ToLowerInvariant(),
            request.Endpoint.Trim(),
            request.CredentialRef,
            MiningStatusDbValues.FromDbValue(request.Status),
            timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return new LlmProviderDto(
            provider.Id,
            provider.Name,
            provider.Endpoint,
            provider.CredentialRef,
            provider.Status.ToDbValue(),
            null,
            null,
            null,
            null,
            [],
            provider.CreatedAt,
            provider.UpdatedAt);
    }
}

public sealed record UpsertDepositAccountCommand(
    Guid LlmProviderId,
    Guid CoinId,
    string Network,
    string DepositAddress,
    string Status) : IRequest<LlmProviderDepositAccountDto>;

public sealed class UpsertDepositAccountCommandValidator : AbstractValidator<UpsertDepositAccountCommand>
{
    public UpsertDepositAccountCommandValidator()
    {
        RuleFor(command => command.LlmProviderId).NotEmpty();
        RuleFor(command => command.CoinId).NotEmpty();
        RuleFor(command => command.Network).NotEmpty().MaximumLength(64);
        RuleFor(command => command.DepositAddress).NotEmpty().MaximumLength(256);
        RuleFor(command => command.Status)
            .Must(value => MiningStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");
    }
}

internal sealed class UpsertDepositAccountCommandHandler(
    ILlmProviderRepository providers,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpsertDepositAccountCommand, LlmProviderDepositAccountDto>
{
    public async Task<LlmProviderDepositAccountDto> Handle(
        UpsertDepositAccountCommand request,
        CancellationToken cancellationToken)
    {
        if (await providers.GetProviderByIdAsync(request.LlmProviderId, cancellationToken) is null)
        {
            throw new NotFoundException("LLM provider not found.");
        }

        var coin = await coins.GetByIdAsync(request.CoinId, cancellationToken)
            ?? throw new NotFoundException("Coin not found.");

        var network = request.Network.Trim();
        var status = MiningStatusDbValues.FromDbValue(request.Status);
        var now = timeProvider.GetUtcNow();

        var account = await providers.GetDepositAccountAsync(
            request.LlmProviderId,
            request.CoinId,
            network,
            cancellationToken);

        if (account is null)
        {
            account = new LlmProviderDepositAccount(
                Guid.NewGuid(),
                request.LlmProviderId,
                request.CoinId,
                network,
                request.DepositAddress.Trim(),
                now);

            providers.AddDepositAccount(account);
        }
        else
        {
            account.Update(request.DepositAddress.Trim(), status, now);
        }

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return new LlmProviderDepositAccountDto(
            account.Id,
            account.CoinId,
            coin.Code,
            account.Network,
            account.DepositAddress,
            account.Status.ToDbValue());
    }
}
