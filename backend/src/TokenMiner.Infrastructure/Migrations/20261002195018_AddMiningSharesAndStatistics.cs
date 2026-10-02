using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddMiningSharesAndStatistics : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "mining_statistics",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    period = table.Column<string>(type: "character varying(16)", maxLength: 16, nullable: false),
                    amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    usd_value = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    calculated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_mining_statistics", x => x.id);
                    table.ForeignKey(
                        name: "fk_mining_statistics_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_mining_statistics_user_hardware_user_hardware_id",
                        column: x => x.user_hardware_id,
                        principalTable: "user_hardware",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_mining_statistics_users_user_id",
                        column: x => x.user_id,
                        principalTable: "users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "service_request_nonces",
                columns: table => new
                {
                    service_id = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    nonce = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    expires_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_service_request_nonces", x => new { x.service_id, x.nonce });
                });

            migrationBuilder.CreateTable(
                name: "user_mining_shares",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_id = table.Column<Guid>(type: "uuid", nullable: false),
                    user_hardware_miner_id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    share_identifier = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: false),
                    worker_identifier = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    job_id = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    nonce = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    extranonce = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    difficulty = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    target = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    result_hash = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    share_timestamp = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    coin_value = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    approx_usd_value_at_time = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    pool_response = table.Column<string>(type: "jsonb", nullable: true),
                    reward_status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    reason = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    processed_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_user_mining_shares", x => x.id);
                    table.ForeignKey(
                        name: "fk_user_mining_shares_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_user_mining_shares_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_user_mining_shares_user_hardware_miners_user_hardware_miner",
                        column: x => x.user_hardware_miner_id,
                        principalTable: "user_hardware_miners",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_user_mining_shares_user_hardware_user_hardware_id",
                        column: x => x.user_hardware_id,
                        principalTable: "user_hardware",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "fk_user_mining_shares_users_user_id",
                        column: x => x.user_id,
                        principalTable: "users",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "ix_mining_statistics_coin_id",
                table: "mining_statistics",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_mining_statistics_user_hardware_id_coin_id_period",
                table: "mining_statistics",
                columns: new[] { "user_hardware_id", "coin_id", "period" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_mining_statistics_user_id",
                table: "mining_statistics",
                column: "user_id");

            migrationBuilder.CreateIndex(
                name: "ix_service_request_nonces_expires_at",
                table: "service_request_nonces",
                column: "expires_at");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_coin_id",
                table: "user_mining_shares",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_pool_id_share_identifier",
                table: "user_mining_shares",
                columns: new[] { "pool_id", "share_identifier" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_reward_status_created_at",
                table: "user_mining_shares",
                columns: new[] { "reward_status", "created_at" });

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_user_hardware_id",
                table: "user_mining_shares",
                column: "user_hardware_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_user_hardware_miner_id",
                table: "user_mining_shares",
                column: "user_hardware_miner_id");

            migrationBuilder.CreateIndex(
                name: "ix_user_mining_shares_user_id_created_at",
                table: "user_mining_shares",
                columns: new[] { "user_id", "created_at" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "mining_statistics");

            migrationBuilder.DropTable(
                name: "service_request_nonces");

            migrationBuilder.DropTable(
                name: "user_mining_shares");
        }
    }
}
