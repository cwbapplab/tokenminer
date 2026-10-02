using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Application.Mining.Shares;

public sealed record RecordShareCommand(
    Guid UserId,
    Guid UserHardwareId,
    Guid PoolId,
    Guid CoinId,
    string ShareIdentifier,
    string? JobId,
    string? Nonce,
    string? Extranonce,
    decimal? Difficulty,
    string? Target,
    string? ResultHash,
    DateTimeOffset ShareTimestamp,
    decimal? CoinValue,
    string? PoolResponse) : IRequest<RecordedShareDto>;

public sealed class RecordShareCommandValidator : AbstractValidator<RecordShareCommand>
{
    public RecordShareCommandValidator()
    {
        RuleFor(x => x.UserId).NotEmpty();
        RuleFor(x => x.UserHardwareId).NotEmpty();
        RuleFor(x => x.PoolId).NotEmpty();
        RuleFor(x => x.CoinId).NotEmpty();
        RuleFor(x => x.ShareIdentifier).NotEmpty().MaximumLength(256);
        RuleFor(x => x.JobId).MaximumLength(128);
        RuleFor(x => x.Nonce).MaximumLength(128);
        RuleFor(x => x.Extranonce).MaximumLength(128);
        RuleFor(x => x.Target).MaximumLength(128);
        RuleFor(x => x.ResultHash).MaximumLength(256);
        RuleFor(x => x.CoinValue).GreaterThanOrEqualTo(0).When(x => x.CoinValue is not null);
    }
}

/// <summary>
/// Records a pool-accepted share. Idempotent on (pool, share identifier): replaying the same
/// share returns the stored row instead of inserting a second one.
/// </summary>
internal sealed class RecordShareCommandHandler(
    IShareRepository shares,
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IPoolRepository pools,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<RecordShareCommand, RecordedShareDto>
{
    public async Task<RecordedShareDto> Handle(RecordShareCommand request, CancellationToken cancellationToken)
    {
        var existing = await shares.GetByPoolAndIdentifierAsync(
            request.PoolId,
            request.ShareIdentifier,
            cancellationToken);

        if (existing is not null)
        {
            return existing.ToDto(alreadyRecorded: true);
        }

        var device = await hardware.GetByIdAsync(request.UserHardwareId, cancellationToken);
        if (device is null || device.UserId != request.UserId)
        {
            throw new BadRequestException("The user/hardware combination does not exist.");
        }

        if (!await pools.SupportsCoinAsync(request.PoolId, request.CoinId, cancellationToken))
        {
            throw new BadRequestException("The pool does not support the requested coin.");
        }

        // The share must belong to the device's live session, which supplies the worker identity.
        var session = await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken)
            ?? throw new BadRequestException("There is no active mining session for this device.");

        var coin = await coins.GetByIdAsync(request.CoinId, cancellationToken)
            ?? throw new BadRequestException("Coin not found.");

        var now = timeProvider.GetUtcNow();

        var share = new UserMiningShare(
            Guid.NewGuid(),
            request.UserId,
            device.Id,
            session.Id,
            request.PoolId,
            request.CoinId,
            request.ShareIdentifier,
            session.WorkerIdentifier,
            request.JobId,
            request.Nonce,
            request.Extranonce,
            request.Difficulty,
            request.Target,
            request.ResultHash,
            request.ShareTimestamp,
            request.CoinValue,
            coin.LastKnownUsdValue,
            request.PoolResponse,
            now);

        shares.Add(share);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return share.ToDto();
    }
}
