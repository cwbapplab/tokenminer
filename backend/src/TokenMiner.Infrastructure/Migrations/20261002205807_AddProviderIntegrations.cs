using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddProviderIntegrations : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "llm_providers",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    name = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    endpoint = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: false),
                    credential_ref = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_llm_providers", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "llm_provider_deposit_accounts",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    llm_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    deposit_address = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_llm_provider_deposit_accounts", x => x.id);
                    table.ForeignKey(
                        name: "fk_llm_provider_deposit_accounts_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_llm_provider_deposit_accounts_llm_providers_llm_provider_id",
                        column: x => x.llm_provider_id,
                        principalTable: "llm_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "llm_provider_models",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    llm_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    model_id = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: false),
                    name = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: true),
                    input_cost = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    output_cost = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    cached_input_cost = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    currency = table.Column<string>(type: "character varying(8)", maxLength: 8, nullable: false),
                    context_length = table.Column<int>(type: "integer", nullable: true),
                    capabilities = table.Column<string>(type: "jsonb", nullable: true),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    last_synced_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_llm_provider_models", x => x.id);
                    table.ForeignKey(
                        name: "fk_llm_provider_models_llm_providers_llm_provider_id",
                        column: x => x.llm_provider_id,
                        principalTable: "llm_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "provider_balance_snapshots",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    llm_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    balance = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    reserved_balance = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    total_balance = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    checked_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_provider_balance_snapshots", x => x.id);
                    table.ForeignKey(
                        name: "fk_provider_balance_snapshots_llm_providers_llm_provider_id",
                        column: x => x.llm_provider_id,
                        principalTable: "llm_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "provider_deposits",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    llm_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    address = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: false),
                    amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    transaction_hash = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    confirmations = table.Column<int>(type: "integer", nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    provider_credit_before = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    provider_credit_after = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    idempotency_key = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: false),
                    error = table.Column<string>(type: "character varying(1024)", maxLength: 1024, nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    confirmed_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_provider_deposits", x => x.id);
                    table.ForeignKey(
                        name: "fk_provider_deposits_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_provider_deposits_llm_providers_llm_provider_id",
                        column: x => x.llm_provider_id,
                        principalTable: "llm_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "ix_llm_provider_deposit_accounts_coin_id",
                table: "llm_provider_deposit_accounts",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_llm_provider_deposit_accounts_llm_provider_id_coin_id_netwo",
                table: "llm_provider_deposit_accounts",
                columns: new[] { "llm_provider_id", "coin_id", "network" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_llm_provider_models_llm_provider_id_model_id",
                table: "llm_provider_models",
                columns: new[] { "llm_provider_id", "model_id" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_llm_providers_name",
                table: "llm_providers",
                column: "name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_provider_balance_snapshots_llm_provider_id_checked_at",
                table: "provider_balance_snapshots",
                columns: new[] { "llm_provider_id", "checked_at" });

            migrationBuilder.CreateIndex(
                name: "ix_provider_deposits_coin_id",
                table: "provider_deposits",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_provider_deposits_idempotency_key",
                table: "provider_deposits",
                column: "idempotency_key",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_provider_deposits_llm_provider_id",
                table: "provider_deposits",
                column: "llm_provider_id");

            migrationBuilder.CreateIndex(
                name: "ix_provider_deposits_status_updated_at",
                table: "provider_deposits",
                columns: new[] { "status", "updated_at" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "llm_provider_deposit_accounts");

            migrationBuilder.DropTable(
                name: "llm_provider_models");

            migrationBuilder.DropTable(
                name: "provider_balance_snapshots");

            migrationBuilder.DropTable(
                name: "provider_deposits");

            migrationBuilder.DropTable(
                name: "llm_providers");
        }
    }
}
