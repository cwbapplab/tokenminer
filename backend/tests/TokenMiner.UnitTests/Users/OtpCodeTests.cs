using FluentAssertions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Users;

public sealed class OtpCodeTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static OtpCode CreateCode(int maxAttempts = 3, TimeSpan? lifetime = null) =>
        new(
            Guid.NewGuid(),
            Guid.NewGuid(),
            OtpPurpose.Activation,
            "hash",
            Now.Add(lifetime ?? TimeSpan.FromMinutes(10)),
            maxAttempts,
            Now);

    [Fact]
    public void NewCode_IsUsable()
    {
        var code = CreateCode();

        code.IsUsable(Now).Should().BeTrue();
        code.HasAttemptsRemaining.Should().BeTrue();
    }

    [Fact]
    public void ExpiredCode_IsNotUsable()
    {
        var code = CreateCode(lifetime: TimeSpan.FromMinutes(5));

        code.IsUsable(Now.AddMinutes(6)).Should().BeFalse();
    }

    [Fact]
    public void Code_StopsBeingUsableOnceAttemptsAreExhausted()
    {
        var code = CreateCode(maxAttempts: 3);

        code.RegisterFailedAttempt();
        code.RegisterFailedAttempt();
        code.IsUsable(Now).Should().BeTrue();

        code.RegisterFailedAttempt();

        code.Attempts.Should().Be(3);
        code.HasAttemptsRemaining.Should().BeFalse();
        code.IsUsable(Now).Should().BeFalse();
    }

    [Fact]
    public void ConsumedCode_IsNotUsable()
    {
        var code = CreateCode();

        code.Consume(Now);

        code.IsConsumed.Should().BeTrue();
        code.IsUsable(Now).Should().BeFalse();
    }
}

public sealed class EmailVerificationTokenTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    [Fact]
    public void NewToken_IsUsable()
    {
        var token = new EmailVerificationToken(Guid.NewGuid(), Guid.NewGuid(), "hash", Now.AddHours(24), Now);

        token.IsUsable(Now).Should().BeTrue();
    }

    [Fact]
    public void ExpiredToken_IsNotUsable()
    {
        var token = new EmailVerificationToken(Guid.NewGuid(), Guid.NewGuid(), "hash", Now.AddHours(1), Now);

        token.IsUsable(Now.AddHours(2)).Should().BeFalse();
    }

    [Fact]
    public void UsedToken_IsNotUsable()
    {
        var token = new EmailVerificationToken(Guid.NewGuid(), Guid.NewGuid(), "hash", Now.AddHours(24), Now);

        token.MarkUsed(Now);

        token.UsedAt.Should().Be(Now);
        token.IsUsable(Now).Should().BeFalse();
    }
}
