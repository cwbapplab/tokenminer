using FluentAssertions;
using TokenMiner.Application.Authentication;
using TokenMiner.Infrastructure.Authentication;
using Xunit;

namespace TokenMiner.UnitTests.Authentication;

public sealed class OtpServiceTests
{
    private static OtpService CreateService(int length = 6) =>
        new(new AuthOptions
        {
            Jwt = new AuthOptions.JwtOptions { SigningKey = "unit-test-signing-key-that-is-long-enough" },
            Otp = new AuthOptions.OtpOptions { Length = length, HmacKey = "unit-test-otp-hmac-key" },
        });

    [Fact]
    public void GenerateCode_ProducesRequestedNumberOfDigits()
    {
        var service = CreateService(length: 6);

        var code = service.GenerateCode();

        code.Should().HaveLength(6);
        code.Should().MatchRegex("^[0-9]{6}$");
    }

    [Fact]
    public void GenerateCode_PadsShortValues()
    {
        // Over many draws the code must never be shorter than the configured length.
        var service = CreateService(length: 6);

        var codes = Enumerable.Range(0, 500).Select(_ => service.GenerateCode()).ToList();

        codes.Should().OnlyContain(code => code.Length == 6);
    }

    [Fact]
    public void Hash_DoesNotContainTheCode()
    {
        var service = CreateService();

        var code = service.GenerateCode();
        var hash = service.Hash(code);

        hash.Should().NotBe(code);
        hash.Should().NotContain(code);
    }

    [Fact]
    public void Hash_IsDeterministic()
    {
        var service = CreateService();

        service.Hash("123456").Should().Be(service.Hash("123456"));
    }

    [Fact]
    public void Verify_AcceptsTheMatchingCode()
    {
        var service = CreateService();
        var code = service.GenerateCode();

        service.Verify(code, service.Hash(code)).Should().BeTrue();
    }

    [Fact]
    public void Verify_RejectsADifferentCode()
    {
        var service = CreateService();

        service.Verify("000000", service.Hash("111111")).Should().BeFalse();
    }

    [Fact]
    public void Verify_RejectsATamperedHash()
    {
        var service = CreateService();
        var code = service.GenerateCode();
        var hash = service.Hash(code);

        var tampered = (hash[0] == '0' ? '1' : '0') + hash[1..];

        service.Verify(code, tampered).Should().BeFalse();
    }

    [Fact]
    public void Verify_RejectsHashOfDifferentLength()
    {
        var service = CreateService();

        service.Verify("123456", "ABCD").Should().BeFalse();
    }
}
