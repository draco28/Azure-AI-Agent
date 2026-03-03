# IT Security Policy - TechCorp Inc.

## 1. Password Requirements

All employees must follow these password standards:

- Minimum 12 characters with uppercase, lowercase, numbers, and special characters
- Passwords must be changed every 90 days
- Previous 10 passwords cannot be reused
- Multi-factor authentication (MFA) is mandatory for all systems

Accounts are locked after 5 consecutive failed login attempts. Contact IT Help Desk (ext. 4357) to unlock.

## 2. Device Management

### 2.1 Company Devices

All company-issued laptops must have:

- Full disk encryption (BitLocker for Windows, FileVault for macOS)
- Endpoint Detection and Response (EDR) software installed
- Automatic OS updates enabled
- Company VPN configured for remote access

Lost or stolen devices must be reported to IT Security within 1 hour at security@techcorp.com.

### 2.2 Personal Devices (BYOD)

Personal devices may access company email and calendar only through the approved MDM (Mobile Device Management) application. Personal devices are NOT permitted to:

- Store company documents locally
- Access source code repositories
- Connect to production systems

## 3. Data Classification

TechCorp classifies data into four levels:

| Level | Examples | Handling |
|-------|----------|----------|
| Public | Marketing materials, blog posts | No restrictions |
| Internal | Meeting notes, project plans | Company network only |
| Confidential | Financial reports, HR records | Encrypted storage, need-to-know access |
| Restricted | Customer PII, trade secrets | Encrypted + audit logging, executive approval required |

Sending Confidential or Restricted data via personal email is a terminable offense.

## 4. Incident Response

If you suspect a security incident:

1. Do NOT attempt to investigate on your own
2. Immediately contact the Security Operations Center (SOC) at soc@techcorp.com or ext. 9111
3. Preserve any evidence (do not delete emails, logs, or files)
4. Document what you observed with timestamps

The SOC team will respond within 15 minutes during business hours and 1 hour outside business hours. All incidents are tracked in the SecOps dashboard with a unique incident ID.

## 5. Software Installation

Only IT-approved software may be installed on company devices. The approved software catalog is available on the IT Portal at https://itportal.techcorp.internal.

Requests for new software must be submitted through the IT Service Desk with:
- Business justification
- Security review (for software handling company data)
- Manager approval

Shadow IT (unauthorized software or cloud services) is prohibited and monitored.

## 6. Network Security

- All internet traffic is routed through the corporate proxy with SSL inspection
- Guest Wi-Fi network ("TechCorp-Guest") is isolated from corporate network
- VPN is required for all remote access to internal systems
- Port scanning or network enumeration on corporate networks is prohibited without written authorization from the CISO

## 7. Compliance Training

All employees must complete annual security awareness training by December 31st each year. New hires must complete training within 30 days of start date. Failure to complete training results in system access suspension until completion.

Training covers: phishing identification, social engineering, data handling, incident reporting, and physical security.
