using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Commands;

public sealed record VerifyOtpCommand(string Email, string Code, string? Ip, string? UserAgent)
    : IRequest<AuthTokens>;

public sealed class VerifyOtpCommandValidator : AbstractValidator<VerifyOtpCommand>
{
    public VerifyOtpCommandValidator()
    {
        RuleFor(x => x.Email).NotEmpty().EmailAddress().MaximumLength(320);
        RuleFor(x => x.Code).NotEmpty().Length(4, 10).Matches("^[0-9]+$")
            .WithMessage("Code must be numeric.");
    }
}

/// <summary>
/// Verifies an activation code, then activates the account and issues tokens. A successful
/// code also confirms the email, since possession of the inbox is what the code proves.
/// </summary>
internal sealed class VerifyOtpCommandHandler(
    IUserRepository users,
    IOtpStore otpStore,
    IOtpService otpService,
    IAuthTokenIssuer tokenIssuer,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<VerifyOtpCommand, AuthTokens>
{
    public async Task<AuthTokens> Handle(VerifyOtpCommand request, CancellationToken cancellationToken)
    {
        var user = await users.GetByNormalizedEmailAsync(
            EmailNormalizer.Normalize(request.Email),
            cancellationToken);

        if (user is null)
        {
            throw new UnauthorizedException("Invalid email or code.");
        }

        var now = timeProvider.GetUtcNow();

        var otp = await otpStore.GetLatestUsableAsync(user.Id, OtpPurpose.Activation, now, cancellationToken);
        if (otp is null)
        {
            throw new UnauthorizedException("Invalid or expired code.");
        }

        if (!otpService.Verify(request.Code, otp.CodeHash))
        {
            otp.RegisterFailedAttempt();
            await unitOfWork.SaveChangesAsync(cancellationToken);

            throw new UnauthorizedException("Invalid or expired code.");
        }

        otp.Consume(now);
        user.ConfirmEmail(now);
        user.Activate(now);

        var roles = await users.GetRoleNamesAsync(user.Id, cancellationToken);
        var issued = await tokenIssuer.IssueAsync(user, roles, request.Ip, request.UserAgent, cancellationToken);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return issued.Tokens;
    }
}
