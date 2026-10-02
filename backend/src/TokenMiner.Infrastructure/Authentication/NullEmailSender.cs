using Microsoft.Extensions.Logging;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Development email sender. Writes the verification link and activation code to the log
/// so the flow is testable without an SMTP server.
/// </summary>
internal sealed class NullEmailSender(ILogger<NullEmailSender> logger) : IEmailSender
{
    public Task SendEmailVerificationAsync(string email, string token, CancellationToken cancellationToken)
    {
        logger.LogInformation("Email verification for {Email} - token: {Token}", email, token);
        return Task.CompletedTask;
    }

    public Task SendActivationOtpAsync(string email, string code, CancellationToken cancellationToken)
    {
        logger.LogInformation("Activation code for {Email} - code: {Code}", email, code);
        return Task.CompletedTask;
    }
}
