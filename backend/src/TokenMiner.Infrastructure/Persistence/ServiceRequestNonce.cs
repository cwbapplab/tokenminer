namespace TokenMiner.Infrastructure.Persistence;

/// <summary>
/// A nonce already used by a signed service request. Purely an infrastructure record used for
/// replay protection, which is why it does not live in the domain model.
/// </summary>
internal sealed class ServiceRequestNonce
{
    private ServiceRequestNonce()
    {
    }

    public ServiceRequestNonce(string serviceId, string nonce, DateTimeOffset expiresAt, DateTimeOffset now)
    {
        ServiceId = serviceId;
        Nonce = nonce;
        ExpiresAt = expiresAt;
        CreatedAt = now;
    }

    public string ServiceId { get; private set; } = null!;

    public string Nonce { get; private set; } = null!;

    public DateTimeOffset ExpiresAt { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }
}
