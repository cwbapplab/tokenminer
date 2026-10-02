using Google.Apis.Auth;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Validates a Google ID token against Google's signing keys, requiring the configured
/// desktop client id as the audience.
/// </summary>
internal sealed class GoogleTokenValidator(AuthOptions options) : IGoogleTokenValidator
{
    public async Task<GoogleUserInfo?> ValidateAsync(string idToken, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(options.Google.DesktopClientId))
        {
            return null;
        }

        try
        {
            var payload = await GoogleJsonWebSignature.ValidateAsync(
                idToken,
                new GoogleJsonWebSignature.ValidationSettings
                {
                    Audience = [options.Google.DesktopClientId],
                });

            return new GoogleUserInfo(payload.Subject, payload.Email, payload.EmailVerified, payload.Name);
        }
        catch (InvalidJwtException)
        {
            return null;
        }
    }
}
