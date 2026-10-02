using System.Collections.Concurrent;
using TokenMiner.Application.Authentication.Abstractions;

namespace TokenMiner.IntegrationTests;

/// <summary>
/// Captures the codes and links the API would have emailed, so tests can complete
/// activation without an SMTP server.
/// </summary>
public sealed class CapturingEmailSender : IEmailSender
{
    private readonly ConcurrentDictionary<string, List<string>> _activationCodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, List<string>> _verificationTokens = new(StringComparer.OrdinalIgnoreCase);

    public Task SendEmailVerificationAsync(string email, string token, CancellationToken cancellationToken)
    {
        _verificationTokens.GetOrAdd(email, _ => []).Add(token);
        return Task.CompletedTask;
    }

    public Task SendActivationOtpAsync(string email, string code, CancellationToken cancellationToken)
    {
        _activationCodes.GetOrAdd(email, _ => []).Add(code);
        return Task.CompletedTask;
    }

    public string LatestActivationCode(string email) =>
        _activationCodes.TryGetValue(email, out var codes) && codes.Count > 0
            ? codes[^1]
            : throw new InvalidOperationException($"No activation code was sent to '{email}'.");

    public string LatestVerificationToken(string email) =>
        _verificationTokens.TryGetValue(email, out var tokens) && tokens.Count > 0
            ? tokens[^1]
            : throw new InvalidOperationException($"No verification token was sent to '{email}'.");
}
