using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record LogoutCommand(string RefreshToken) : IRequest;

public sealed class LogoutCommandValidator : AbstractValidator<LogoutCommand>
{
    public LogoutCommandValidator()
    {
        RuleFor(x => x.RefreshToken).NotEmpty();
    }
}

/// <summary>Revokes a refresh token. Idempotent: an unknown token is treated as success.</summary>
internal sealed class LogoutCommandHandler(
    IRefreshTokenStore refreshTokens,
    ISecureTokenService secureTokenService,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<LogoutCommand>
{
    public async Task Handle(LogoutCommand request, CancellationToken cancellationToken)
    {
        var token = await refreshTokens.GetByHashAsync(
            secureTokenService.Hash(request.RefreshToken),
            cancellationToken);

        if (token is null || token.IsRevoked)
        {
            return;
        }

        token.Revoke("logout", timeProvider.GetUtcNow());

        await unitOfWork.SaveChangesAsync(cancellationToken);
    }
}
