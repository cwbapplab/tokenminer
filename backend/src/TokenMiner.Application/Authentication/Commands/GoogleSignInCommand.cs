using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Commands;

/// <summary>
/// Signs in with a Google ID token obtained by the desktop client's native sign-in flow.
/// The API never participates in a browser redirect; it only validates the token audience.
/// </summary>
public sealed record GoogleSignInCommand(string IdToken, string? Ip, string? UserAgent)
    : IRequest<GoogleSignInResult>;

public sealed class GoogleSignInCommandValidator : AbstractValidator<GoogleSignInCommand>
{
    public GoogleSignInCommandValidator()
    {
        RuleFor(x => x.IdToken).NotEmpty();
    }
}

internal sealed class GoogleSignInCommandHandler(
    IUserRepository users,
    IOtpStore otpStore,
    IOtpService otpService,
    IEmailSender emailSender,
    IGoogleTokenValidator googleTokenValidator,
    IAuthTokenIssuer tokenIssuer,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    AuthOptions options) : IRequestHandler<GoogleSignInCommand, GoogleSignInResult>
{
    public async Task<GoogleSignInResult> Handle(GoogleSignInCommand request, CancellationToken cancellationToken)
    {
        var googleUser = await googleTokenValidator.ValidateAsync(request.IdToken, cancellationToken)
            ?? throw new UnauthorizedException("Invalid Google token.");

        if (!googleUser.EmailVerified)
        {
            throw new UnauthorizedException("Google account email is not verified.");
        }

        var now = timeProvider.GetUtcNow();

        var user = await users.GetByGoogleSubAsync(googleUser.Subject, cancellationToken)
            ?? await users.GetByNormalizedEmailAsync(EmailNormalizer.Normalize(googleUser.Email), cancellationToken);

        var isNewUser = user is null;

        if (user is null)
        {
            user = new User(
                Guid.NewGuid(),
                googleUser.Email.Trim(),
                EmailNormalizer.Normalize(googleUser.Email),
                passwordHash: null,
                googleUser.Name,
                now);

            user.LinkGoogle(googleUser.Subject, now);
            user.ConfirmEmail(now);

            var defaultRole = await users.GetRoleByNameAsync(Role.User, cancellationToken)
                ?? throw new InvalidOperationException($"Default role '{Role.User}' is not seeded.");

            users.Add(user);
            users.AddUserRole(new UserRole(user.Id, defaultRole.Id));
        }
        else
        {
            user.LinkGoogle(googleUser.Subject, now);

            // Google has verified the address, which is what the email-validation step would prove.
            user.ConfirmEmail(now);
        }

        if (options.RequireOtpForOauth && !user.IsActive)
        {
            await otpStore.InvalidateOutstandingAsync(user.Id, OtpPurpose.Activation, now, cancellationToken);

            var code = otpService.GenerateCode();
            otpStore.Add(new OtpCode(
                Guid.NewGuid(),
                user.Id,
                OtpPurpose.Activation,
                otpService.Hash(code),
                now.AddMinutes(options.Otp.ExpiryMinutes),
                options.Otp.MaxAttempts,
                now));

            await unitOfWork.SaveChangesAsync(cancellationToken);
            await emailSender.SendActivationOtpAsync(user.Email, code, cancellationToken);

            return GoogleSignInResult.ActivationRequired();
        }

        if (!user.IsActive)
        {
            user.Activate(now);
        }

        // A freshly added user's role assignment is not visible to a database query yet.
        var roles = isNewUser
            ? [Role.User]
            : await users.GetRoleNamesAsync(user.Id, cancellationToken);

        var issued = await tokenIssuer.IssueAsync(user, roles, request.Ip, request.UserAgent, cancellationToken);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return GoogleSignInResult.Success(issued.Tokens);
    }
}
