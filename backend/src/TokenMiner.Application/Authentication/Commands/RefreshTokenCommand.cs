using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record RefreshTokenCommand(string RefreshToken, string? Ip, string? UserAgent)
    : IRequest<AuthTokens>;

public sealed class RefreshTokenCommandValidator : AbstractValidator<RefreshTokenCommand>
{
    public RefreshTokenCommandValidator()
    {
        RuleFor(x => x.RefreshToken).NotEmpty();
    }
}

/// <summary>
/// Rotates a refresh token: the presented token is revoked and a fresh pair issued.
///
/// Presenting an already-revoked token is treated as a possible theft, but the response is
/// deliberately narrow — the session the replay belongs to is what gets shut down. A replay of
/// an old token from one client must not sign out that user's other devices, which is exactly
/// what happens when several clients share an account.
///
/// The rotation itself is a single conditional UPDATE (revoke-where-still-active), so two
/// concurrent refreshes of the same token cannot both win: the loser sees zero rows affected
/// and takes the replay path instead of issuing a second pair.
/// </summary>
internal sealed class RefreshTokenCommandHandler(
    IUserRepository users,
    IRefreshTokenStore refreshTokens,
    ISecureTokenService secureTokenService,
    IAuthTokenIssuer tokenIssuer,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<RefreshTokenCommand, AuthTokens>
{
    public async Task<AuthTokens> Handle(RefreshTokenCommand request, CancellationToken cancellationToken)
    {
        // The rotation is a conditional write that cannot be replayed, so the work runs through the
        // store's execution strategy. Today that strategy does not retry; this keeps the handler
        // correct if one is enabled later.
        var strategy = unitOfWork.CreateExecutionStrategy();
        return await strategy.ExecuteAsync(async () =>
        {
            var now = timeProvider.GetUtcNow();

            var token = await refreshTokens.GetByHashAsync(
                secureTokenService.Hash(request.RefreshToken),
                cancellationToken);

            if (token is null)
            {
                throw new UnauthorizedException("Invalid refresh token.");
            }

            if (token.IsRevoked)
            {
                await CloseReplayedSessionAsync(token, now, cancellationToken);

                throw new UnauthorizedException("Refresh token is no longer valid.");
            }

            if (token.IsExpired(now))
            {
                throw new UnauthorizedException("Refresh token has expired.");
            }

            var user = await users.GetByIdAsync(token.UserId, cancellationToken)
                ?? throw new UnauthorizedException("Invalid refresh token.");

            if (!user.IsActive)
            {
                throw new ForbiddenException("Account is not activated.");
            }

            var roles = await users.GetRoleNamesAsync(user.Id, cancellationToken);
            var issued = await tokenIssuer.IssueAsync(user, roles, request.Ip, request.UserAgent, cancellationToken);

            // Claim the rotation. Two callers racing on the same token both get here; this is the
            // line that lets exactly one through. It stores the new token as part of the same
            // transaction, so losing the race leaves no orphaned pair behind.
            var claimed = await refreshTokens.TryRotateAsync(token.Id, "rotated", now, issued.RefreshTokenId, cancellationToken);
            if (!claimed)
            {
                throw new UnauthorizedException("Refresh token is no longer valid.");
            }

            return issued.Tokens;
        });
    }

    /// <summary>
    /// Records a reuse and closes only the session the replayed token belongs to.
    /// </summary>
    /// <remarks>
    /// The replayed token is already revoked, so the work is to stop the chain it was rotated
    /// into. Following that chain — rather than revoking every active token for the user —
    /// contains a compromised lineage to a single session and leaves the user's other devices
    /// signed in. A token with no recorded successor (issued before rotation tracking) has
    /// nothing downstream to stop, and the session it came from is unreachable anyway since its
    /// client no longer holds a usable token.
    /// </remarks>
    private Task CloseReplayedSessionAsync(
        RefreshToken token,
        DateTimeOffset now,
        CancellationToken cancellationToken) =>
        refreshTokens.RevokeSessionChainAsync(token.Id, "reuse_detected", now, cancellationToken);
}
