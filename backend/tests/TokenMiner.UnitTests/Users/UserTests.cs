using FluentAssertions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;
using Xunit;

namespace TokenMiner.UnitTests.Users;

public sealed class UserTests
{
    private static readonly DateTimeOffset Now = new(2026, 1, 1, 12, 0, 0, TimeSpan.Zero);

    private static User CreateUser() =>
        new(Guid.NewGuid(), "user@tokenminer.local", "USER@TOKENMINER.LOCAL", "hash", "Test", Now);

    [Fact]
    public void NewUser_StartsPendingActivation()
    {
        var user = CreateUser();

        user.Status.Should().Be(UserStatus.PendingActivation);
        user.IsActive.Should().BeFalse();
        user.EmailConfirmedAt.Should().BeNull();
        user.CreatedAt.Should().Be(Now);
        user.UpdatedAt.Should().Be(Now);
    }

    [Fact]
    public void ConfirmEmail_KeepsTheFirstConfirmationTime()
    {
        var user = CreateUser();

        user.ConfirmEmail(Now);
        user.ConfirmEmail(Now.AddDays(1));

        user.EmailConfirmedAt.Should().Be(Now);
    }

    [Fact]
    public void Activate_MarksActiveAndBumpsUpdatedAt()
    {
        var user = CreateUser();
        var later = Now.AddMinutes(5);

        user.Activate(later);

        user.Status.Should().Be(UserStatus.Active);
        user.IsActive.Should().BeTrue();
        user.UpdatedAt.Should().Be(later);
    }

    [Fact]
    public void LinkGoogle_DoesNotOverwriteAnExistingSubject()
    {
        var user = CreateUser();

        user.LinkGoogle("google-subject-1", Now);
        user.LinkGoogle("google-subject-2", Now.AddMinutes(1));

        user.GoogleSub.Should().Be("google-subject-1");
    }

    [Fact]
    public void Suspend_MarksSuspended()
    {
        var user = CreateUser();
        user.Activate(Now.AddSeconds(1));

        user.Suspend(Now.AddSeconds(2));

        user.Status.Should().Be(UserStatus.Suspended);
        user.IsActive.Should().BeFalse();
    }
}
