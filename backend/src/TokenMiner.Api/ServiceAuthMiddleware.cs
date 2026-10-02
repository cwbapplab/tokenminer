using System.Security.Cryptography;
using TokenMiner.Application.Mining;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Services;

namespace TokenMiner.Api;

/// <summary>
/// Authentication for machine-to-machine endpoints. Verifies an HMAC signature over
/// (method, path, timestamp, nonce, body hash), enforces a clock-skew window and rejects any
/// nonce it has already seen.
/// </summary>
/// <remarks>
/// This runs as middleware rather than an endpoint filter on purpose: minimal APIs bind the
/// handler's arguments before filters execute, so by filter time the request body has already
/// been consumed and its hash can no longer be computed.
/// </remarks>
internal sealed class ServiceAuthMiddleware(RequestDelegate next)
{
    private const string ServiceIdHeader = "X-Service-Id";
    private const string TimestampHeader = "X-Timestamp";
    private const string NonceHeader = "X-Nonce";
    private const string SignatureHeader = "X-Signature";

    public async Task InvokeAsync(
        HttpContext context,
        MiningOptions options,
        IServiceNonceStore nonces,
        TimeProvider timeProvider,
        ILogger<ServiceAuthMiddleware> logger)
    {
        var request = context.Request;

        if (string.IsNullOrWhiteSpace(options.ServiceSharedSecret))
        {
            logger.LogError("Mining:ServiceSharedSecret is not configured; internal endpoints are unavailable.");
            await WriteProblemAsync(
                context,
                StatusCodes.Status503ServiceUnavailable,
                "Service authentication is not configured.");
            return;
        }

        var serviceId = request.Headers[ServiceIdHeader].ToString();
        var timestamp = request.Headers[TimestampHeader].ToString();
        var nonce = request.Headers[NonceHeader].ToString();
        var signature = request.Headers[SignatureHeader].ToString();

        if (string.IsNullOrWhiteSpace(serviceId)
            || string.IsNullOrWhiteSpace(timestamp)
            || string.IsNullOrWhiteSpace(nonce)
            || string.IsNullOrWhiteSpace(signature))
        {
            await WriteUnauthorizedAsync(context, "Missing service authentication headers.");
            return;
        }

        if (!long.TryParse(timestamp, out var unixSeconds))
        {
            await WriteUnauthorizedAsync(context, "Malformed timestamp.");
            return;
        }

        var now = timeProvider.GetUtcNow();

        if ((now - DateTimeOffset.FromUnixTimeSeconds(unixSeconds)).Duration()
            > TimeSpan.FromSeconds(options.ServiceClockSkewSeconds))
        {
            await WriteUnauthorizedAsync(context, "Timestamp is outside the allowed window.");
            return;
        }

        var bodyHash = await ComputeBodyHashAsync(request, context.RequestAborted);

        if (!ServiceRequestSignature.Verify(
                options.ServiceSharedSecret,
                request.Method,
                request.Path.Value ?? string.Empty,
                timestamp,
                nonce,
                bodyHash,
                signature))
        {
            await WriteUnauthorizedAsync(context, "Invalid signature.");
            return;
        }

        // Registered only once the signature is proven valid, so a forged request cannot burn a nonce.
        var registered = await nonces.TryRegisterAsync(
            serviceId,
            nonce,
            now.AddSeconds(options.ServiceNonceRetentionSeconds),
            context.RequestAborted);

        if (!registered)
        {
            await WriteUnauthorizedAsync(context, "Replayed request.");
            return;
        }

        await next(context);
    }

    /// <summary>
    /// Hashes the raw body, then rewinds so the endpoint can still bind the payload.
    /// </summary>
    private static async Task<string> ComputeBodyHashAsync(HttpRequest request, CancellationToken cancellationToken)
    {
        request.EnableBuffering();
        request.Body.Position = 0;

        var hash = await SHA256.HashDataAsync(request.Body, cancellationToken);

        request.Body.Position = 0;

        return Convert.ToHexString(hash);
    }

    private static Task WriteUnauthorizedAsync(HttpContext context, string detail) =>
        WriteProblemAsync(context, StatusCodes.Status401Unauthorized, detail);

    private static async Task WriteProblemAsync(HttpContext context, int statusCode, string detail)
    {
        var problem = Results.Problem(detail: detail, statusCode: statusCode);

        await problem.ExecuteAsync(context);
    }
}
