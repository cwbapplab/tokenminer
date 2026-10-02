using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;

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
/// Rotates a refresh token. Presenting an already-revoked token is treated as a possible
/// theft: every active token for that user is revoked, forcing a fresh sign-in.
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
        var token = await refreshTokens.GetByHashAsync(
            secureTokenService.Hash(request.RefreshToken),
            cancellationToken);

        if (token is null)
        {
            throw new UnauthorizedException("Invalid refresh token.");
        }

        var now = timeProvider.GetUtcNow();

        if (token.IsRevoked)
        {
            var activeTokens = await refreshTokens.GetActiveByUserAsync(token.UserId, cancellationToken);
            foreach (var activeToken in activeTokens)
            {
                activeToken.Revoke("reuse_detected", now);
            }

            await unitOfWork.SaveChangesAsync(cancellationToken);

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

        token.Revoke("rotated", now, issued.RefreshTokenId);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return issued.Tokens;
    }
}
