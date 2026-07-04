// lightweight_cipher: a small substitution-permutation-network (SPN) block
// cipher core, hand-written for this project as a custom benchmark sized
// for the eFPGA logic-masking use case described in CLAUDE.md -- one small
// critical IP block meant to be embedded whole in a tiny custom fabric, not
// a generic FPGA-benchmark-suite design borrowed at the wrong scale.
//
// A cryptographic core is also the canonical "critical IP" the masking
// technique targets in the first place: if the cipher's structure is
// visible from the GDSII, an attacker recovers it without ever needing the
// bitstream.
//
// Architecture: byte-serial SubBytes (one 8-bit S-box lookup per clock --
// the defining area-saving trait of real lightweight-cipher hardware, e.g.
// PRESENT/GIFT/SPECK-class ciphers), a free wire-level byte rotation for
// diffusion, and a key-XOR with an LFSR-style round-key update. The S-box
// is loaded into a small RAM once after reset (BRAM #1); a 32-entry
// ciphertext history buffer (BRAM #2) records the last 32 outputs. No
// multiplication is used anywhere (DSP usage = 0), matching real
// lightweight-cipher design practice, which avoids multipliers for area.
module lightweight_cipher (
    input clk,
    input reset,
    input start,
    input  [63:0] key_in,
    input  [63:0] plaintext_in,
    input  [4:0]  raddr,
    output reg [63:0] ciphertext_out,
    output reg        done,
    output reg [63:0] stored_ct_out
);

    localparam NUM_ROUNDS = 8;

    reg [7:0]  sbox_mem  [0:255];   // BRAM #1: substitution table
    reg [63:0] ctext_buf [0:31];    // BRAM #2: ciphertext history buffer

    // ---- S-box load (runs once after reset, no multiplier used) ----
    reg [8:0] init_idx;
    reg       loading;

    wire [7:0] li   = init_idx[7:0];
    wire [7:0] rot5 = {li[2:0], li[7:3]};
    wire [7:0] rot3 = {li[4:0], li[7:5]};
    wire [7:0] sbox_init_val = (rot5 ^ rot3) ^ 8'h63;

    // ---- Cipher datapath ----
    reg [63:0] state;
    reg [63:0] subbed;
    reg [63:0] rkey;
    reg [3:0]  round;
    reg [2:0]  byte_idx;
    reg [4:0]  waddr_ctr;
    reg        sub_phase;   // 0 = present S-box address, 1 = capture result
    reg [7:0]  sub_reg;     // registered S-box read result

    reg [2:0] fsm;
    localparam S_IDLE = 0, S_SUB = 1, S_XOR = 2, S_DONE = 3;

    wire [7:0] cur_byte = (byte_idx == 3'd0) ? state[7:0]   :
                          (byte_idx == 3'd1) ? state[15:8]  :
                          (byte_idx == 3'd2) ? state[23:16] :
                          (byte_idx == 3'd3) ? state[31:24] :
                          (byte_idx == 3'd4) ? state[39:32] :
                          (byte_idx == 3'd5) ? state[47:40] :
                          (byte_idx == 3'd6) ? state[55:48] :
                                               state[63:56];

    wire [63:0] permuted = {subbed[7:0], subbed[63:8]};

    always @(posedge clk) begin
        if (reset) begin
            loading        <= 1'b1;
            init_idx       <= 9'd0;
            fsm            <= S_IDLE;
            state          <= 64'd0;
            subbed         <= 64'd0;
            rkey           <= 64'd0;
            round          <= 4'd0;
            byte_idx       <= 3'd0;
            waddr_ctr      <= 5'd0;
            sub_phase      <= 1'b0;
            done           <= 1'b0;
            ciphertext_out <= 64'd0;
        end else if (loading) begin
            sbox_mem[init_idx[7:0]] <= sbox_init_val;
            if (init_idx == 9'd255)
                loading <= 1'b0;
            else
                init_idx <= init_idx + 9'd1;
        end else begin
            done <= 1'b0;
            sub_reg <= sbox_mem[cur_byte];
            case (fsm)
                S_IDLE: begin
                    if (start) begin
                        state     <= plaintext_in;
                        rkey      <= key_in;
                        round     <= 4'd0;
                        byte_idx  <= 3'd0;
                        sub_phase <= 1'b0;
                        fsm       <= S_SUB;
                    end
                end
                S_SUB: begin
                    if (sub_phase == 1'b0) begin
                        sub_phase <= 1'b1;
                    end else begin
                        case (byte_idx)
                            3'd0: subbed[7:0]   <= sub_reg;
                            3'd1: subbed[15:8]  <= sub_reg;
                            3'd2: subbed[23:16] <= sub_reg;
                            3'd3: subbed[31:24] <= sub_reg;
                            3'd4: subbed[39:32] <= sub_reg;
                            3'd5: subbed[47:40] <= sub_reg;
                            3'd6: subbed[55:48] <= sub_reg;
                            3'd7: subbed[63:56] <= sub_reg;
                        endcase
                        if (byte_idx == 3'd7) begin
                            fsm <= S_XOR;
                        end else begin
                            byte_idx  <= byte_idx + 3'd1;
                            sub_phase <= 1'b0;
                        end
                    end
                end
                S_XOR: begin
                    state <= permuted ^ rkey;
                    rkey  <= {rkey[31:0], rkey[63:32]} ^ 64'hA5A5A5A5A5A5A5A5;
                    if (round == NUM_ROUNDS - 1) begin
                        fsm <= S_DONE;
                    end else begin
                        round     <= round + 4'd1;
                        byte_idx  <= 3'd0;
                        sub_phase <= 1'b0;
                        fsm       <= S_SUB;
                    end
                end
                S_DONE: begin
                    ciphertext_out       <= state;
                    ctext_buf[waddr_ctr] <= state;
                    waddr_ctr            <= waddr_ctr + 5'd1;
                    done                 <= 1'b1;
                    fsm                  <= S_IDLE;
                end
                default: fsm <= S_IDLE;
            endcase
            stored_ct_out <= ctext_buf[raddr];
        end
    end

endmodule
