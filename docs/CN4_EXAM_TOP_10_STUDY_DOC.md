# CN4 Computer Networks Exam - Top 10 Study Questions

Source question bank: `C:\Users\Ashish\all\Downloads\CN4.1bct.pdf`

This document turns the question bank into a high-yield reading plan. The priority is not to read passively. For each block, prepare one clean answer structure, one diagram where useful, and one solved numerical example.

## 1. Subnetting / VLSM

**Most important question pattern**

Given an IPv4 block and departments with different host counts, divide the network using minimum wastage. For each subnet, list:

- CIDR / subnet mask
- Network address
- Broadcast address
- Usable host range
- Wasted IP addresses

**Why this is high yield**

Subnetting appears repeatedly across the papers and usually carries a full numerical mark block. It is also easy to score if the method is clean.

**Answer framework**

1. Sort departments by descending host requirement.
2. For each department, choose the smallest block where `2^n - 2 >= required hosts`.
3. Allocate largest blocks first from the starting address.
4. For each subnet, write network address, first usable, last usable, broadcast address, subnet mask, and wasted IPs.
5. Reserve /30 blocks for point-to-point links when asked.

**Practice target**

Solve at least two examples from the question bank without looking at the solution. This is the first block to nail.

## 2. IPv4 vs IPv6 and Transition Mechanisms

**Most important question pattern**

Compare IPv4 and IPv6, especially header differences, routing/header handling, address size, fragmentation, security support, and transition mechanisms.

Common transition mechanisms:

- Dual stack
- Tunneling
- Header translation
- 6RD
- ISATAP
- 6to4

**Why this is high yield**

IPv6 appears in many forms: advantages over IPv4, header comparison, address types, auto-configuration, coexistence, tunneling, and migration strategies.

**Answer framework**

1. Start with the core limitation of IPv4: address exhaustion and complex header processing.
2. Compare header size, checksum, fragmentation, options/extension headers, address length, NAT dependence, and security.
3. Explain dual stack as the safest coexistence strategy.
4. Explain tunneling as IPv6 packets carried over IPv4 infrastructure.
5. Explain translation as converting packet headers between IPv4 and IPv6 networks.
6. Add a small diagram if the question asks for a transition strategy.

## 3. Routing Protocols: Distance Vector, Link State, OSPF/RIP

**Most important question pattern**

Compare distance vector and link state routing, then explain one routing protocol or process such as OSPF, RIP, link-state routing table formation, routing loops, DR/BDR election, or shortest path computation.

**Why this is high yield**

Routing is one of the most repeated theory areas. It also combines comparison, process explanation, and diagram-based answers.

**Answer framework**

1. Define routing and routing protocol.
2. Compare distance vector and link state:
   - Information shared
   - Update frequency
   - Algorithm used
   - Convergence speed
   - Loop behavior
   - Example protocols
3. For OSPF/link state:
   - Neighbor discovery
   - Link-state advertisement
   - Link-state database
   - Dijkstra shortest path tree
   - Routing table generation
4. For DR/BDR:
   - Explain why election is needed on broadcast networks.
   - Mention router priority and router ID.

## 4. TCP/UDP, Reliability, Handshake, and Congestion Control

**Most important question pattern**

Explain TCP reliability, TCP header/segment structure, 3-way handshake and termination, UDP header/use cases, and congestion control using leaky bucket and token bucket.

**Why this is high yield**

Transport-layer questions repeat almost every year. The exam often mixes theory with comparison.

**Answer framework**

1. Define transport layer responsibilities: process-to-process delivery, segmentation, reliability, flow control, congestion control, and port addressing.
2. Explain TCP:
   - Connection-oriented
   - Reliable delivery
   - Sequence and acknowledgment numbers
   - Retransmission
   - Flow control
   - 3-way handshake: SYN, SYN-ACK, ACK
3. Explain UDP:
   - Connectionless
   - Low overhead
   - Used for DNS, streaming, VoIP, gaming, and simple request-response traffic.
4. Compare leaky bucket and token bucket:
   - Leaky bucket smooths traffic at a fixed rate.
   - Token bucket allows bursts when tokens are available.

## 5. Network Security: Firewall, RSA, Secure Communication

**Most important question pattern**

Explain firewall/types or secure communication properties, then solve RSA encryption/decryption for a given word.

**Why this is high yield**

Security appears repeatedly, and RSA numerical problems are especially common. This is a scoring block if the steps are memorized and practiced.

**Answer framework**

1. Secure communication properties:
   - Confidentiality
   - Integrity
   - Authentication
   - Non-repudiation
   - Availability
2. Firewall:
   - Define firewall as a security mechanism that filters traffic based on rules.
   - Explain packet-filtering firewall, stateful firewall, proxy/application firewall, and next-generation firewall if needed.
3. RSA steps:
   - Choose primes `p` and `q`.
   - Compute `n = p * q`.
   - Compute `phi(n) = (p - 1)(q - 1)`.
   - Choose public exponent `e` such that `gcd(e, phi(n)) = 1`.
   - Compute private exponent `d` such that `e*d mod phi(n) = 1`.
   - Encrypt: `C = M^e mod n`.
   - Decrypt: `M = C^d mod n`.

**Solved numerical: Encrypt and decrypt `ATTACK` using RSA**

Use exam-friendly letter mapping:

```text
A = 1, B = 2, ..., Z = 26
ATTACK = 1 20 20 1 3 11
```

Choose small RSA values:

```text
p = 3, q = 11
n = p*q = 3*11 = 33
phi(n) = (p - 1)(q - 1) = 2*10 = 20
Choose e = 7 because gcd(7, 20) = 1
Find d such that e*d mod phi(n) = 1
7*d mod 20 = 1
d = 3 because 7*3 = 21 = 1 mod 20
```

Therefore:

```text
Public key = (e, n) = (7, 33)
Private key = (d, n) = (3, 33)
```

Encryption formula:

```text
C = M^e mod n = M^7 mod 33
```

| Letter | M | C = M^7 mod 33 |
|---|---:|---:|
| A | 1 | 1 |
| T | 20 | 26 |
| T | 20 | 26 |
| A | 1 | 1 |
| C | 3 | 9 |
| K | 11 | 11 |

Encrypted message:

```text
1 26 26 1 9 11
```

Decryption formula:

```text
M = C^d mod n = C^3 mod 33
```

| Cipher C | M = C^3 mod 33 | Letter |
|---:|---:|---|
| 1 | 1 | A |
| 26 | 20 | T |
| 26 | 20 | T |
| 1 | 1 | A |
| 9 | 3 | C |
| 11 | 11 | K |

Decrypted message:

```text
ATTACK
```

Exam memory line: convert each letter to a number, encrypt each number using `C = M^e mod n`, then decrypt using `M = C^d mod n`.

## 6. OSI, TCP/IP, and Layered Architecture

**Most important question pattern**

Define protocol/network, explain why layered architecture is used, list the function of each OSI or TCP/IP layer, and compare OSI with TCP/IP.

**Why this is high yield**

This is a common opening question. It is usually straightforward marks if your layer functions and comparison are organized.

**Answer framework**

1. Define protocol as a set of rules for communication between network entities.
2. Explain why layers are used:
   - Reduces complexity
   - Standardizes communication
   - Allows modular design
   - Makes troubleshooting easier
   - Lets one layer change without rewriting the whole system
3. OSI layers:
   - Physical: bits, signals, cables
   - Data Link: framing, MAC addressing, error detection
   - Network: logical addressing and routing
   - Transport: end-to-end delivery, reliability, ports
   - Session: session management
   - Presentation: encryption, compression, format conversion
   - Application: user-facing network services
4. Compare OSI and TCP/IP:
   - OSI has 7 layers; TCP/IP has 4 or 5 layers depending on representation.
   - OSI is a reference model; TCP/IP is a practical protocol suite.
   - TCP/IP combines OSI session, presentation, and application functions.

## 7. Data Link Layer: CRC, ARQ, Framing, CSMA/CD, ALOHA

**Most important question pattern**

Explain error detection/correction, calculate CRC, explain bit stuffing/framing, compare Go-Back-N and Selective Repeat ARQ, and explain ALOHA, slotted ALOHA, CSMA/CD, or CSMA/CA.

**Why this is high yield**

Data link questions repeat in both theory and numerical form. CRC and bit stuffing are especially useful because they are direct scoring questions.

**Answer framework**

1. Data link responsibilities:
   - Framing
   - Physical addressing
   - Error detection and correction
   - Flow control
   - Medium access control
2. CRC steps:
   - Append zeros equal to generator degree.
   - Divide message by generator using modulo-2 division.
   - Append remainder to the original message.
   - Receiver divides received frame by the same generator; non-zero remainder means error.
3. ARQ:
   - Stop-and-Wait: one frame at a time.
   - Go-Back-N: resend from the damaged/lost frame onward.
   - Selective Repeat: resend only damaged/lost frames.
4. Multiple access:
   - Pure ALOHA transmits anytime; collision chance is high.
   - Slotted ALOHA transmits only at slot boundaries.
   - CSMA/CD detects collisions in wired Ethernet.
   - CSMA/CA avoids collisions in wireless LAN.

## 8. Application Layer: DNS, HTTP/HTTPS, FTP, SMTP, POP3, IMAP

**Most important question pattern**

Explain DNS recursive and iterative queries, DNS delegation/records, HTTP/HTTPS browsing, FTP connection process, SMTP working steps, and POP3 vs IMAP.

**Why this is high yield**

Application-layer questions appear repeatedly and are usually diagram-friendly. DNS and email protocols are especially common.

**Answer framework**

1. DNS:
   - Converts domain names to IP addresses.
   - Recursive query: DNS server does the full lookup for the client.
   - Iterative query: each DNS server refers the resolver to the next server.
   - Mention root, TLD, authoritative server, and local resolver.
2. HTTP/HTTPS:
   - HTTP is stateless request-response web communication.
   - HTTPS is HTTP over TLS/SSL for encrypted communication.
3. FTP:
   - Uses a control connection and a data connection.
   - Control commonly uses port 21.
4. Email:
   - SMTP sends mail.
   - POP3 downloads mail, often to one device.
   - IMAP keeps mail synchronized on the server across devices.

## 9. Switching, Multiplexing, and Transmission Media

**Most important question pattern**

Define switching and multiplexing, compare circuit switching and packet switching, explain virtual circuit/datagram switching, and describe guided/unguided transmission media.

**Why this is high yield**

This area appears throughout older and newer papers. It is a strong theory block and often includes comparison tables.

**Answer framework**

1. Switching:
   - Circuit switching reserves a dedicated path before communication.
   - Packet switching divides data into packets and routes each packet through the network.
   - Datagram switching routes each packet independently.
   - Virtual circuit switching establishes a logical path first, then sends packets along it.
2. Multiplexing:
   - Combines multiple signals over one shared medium.
   - Common types: FDM, TDM, WDM, CDM.
3. Transmission media:
   - Guided: twisted pair, coaxial cable, optical fiber.
   - Unguided: radio wave, microwave, infrared, satellite.
4. Optical fiber:
   - High bandwidth
   - Low attenuation
   - Immune to electromagnetic interference
   - Used for long-distance and backbone communication

## 10. Network Devices, LAN Standards, VLAN, and Frame Relay/X.25

**Most important question pattern**

Explain repeaters, hubs, bridges, switches, routers, gateways, VLANs, Ethernet frame structure, token bus/ring, Frame Relay, or X.25 virtual circuit operation.

**Why this is high yield**

These topics frequently appear as short notes or mid-sized theory questions. They are good backup marks after the top numerical blocks.

**Answer framework**

1. Devices:
   - Repeater: regenerates signal at physical layer.
   - Hub: broadcasts frames to all ports.
   - Bridge: connects LAN segments and filters by MAC address.
   - Switch: forwards frames using a MAC address table.
   - Router: forwards packets using IP addresses.
   - Gateway: connects different protocol environments.
2. VLAN:
   - Logically separates a physical LAN into multiple broadcast domains.
   - Improves security, traffic control, and network management.
3. Ethernet frame:
   - Preamble
   - Destination MAC
   - Source MAC
   - Type/length
   - Data
   - FCS
4. Frame Relay/X.25:
   - Both are virtual-circuit based WAN technologies.
   - X.25 includes stronger error control and is slower.
   - Frame Relay is faster and assumes reliable links.

## Reading Order

1. Subnetting / VLSM
2. RSA and firewall/security
3. IPv4 vs IPv6 and transition
4. Routing protocols
5. TCP/UDP and congestion control
6. OSI, TCP/IP, and layered architecture
7. Data link layer: CRC, ARQ, framing, CSMA/CD, ALOHA
8. Application layer: DNS, HTTP/HTTPS, FTP, SMTP, POP3, IMAP
9. Switching, multiplexing, and transmission media
10. Network devices, LAN standards, VLAN, Frame Relay/X.25

## Exam Execution Rule

For each block, prepare:

- One definition opening
- One comparison table
- One step-by-step process
- One diagram where relevant
- One solved numerical example where applicable

Do not reread randomly. Close one block, test yourself, then move to the next block.
