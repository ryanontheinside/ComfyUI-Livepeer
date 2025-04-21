# Livepeer ComfyUI Architecture

This document explains the architecture of the Livepeer ComfyUI integration, particularly focusing on the job handling system.

## Key Components

### LivepeerJobService
- **Purpose**: Centralized service for managing the lifecycle and state of all Livepeer jobs
- **Responsibilities**:
  - Job registration and tracking
  - State management and result storage
  - Node association tracking
  - Automatic cleanup of completed jobs
  - Thread-safe job data access

### LivepeerBase
- **Purpose**: Base class for all Livepeer requester nodes
- **Responsibilities**:
  - Handling API request execution with retry logic
  - Remote API polling for job status
  - Registering new jobs with the service
  - Managing synchronous and asynchronous job execution
  - Error handling and response processing

### LivepeerJobGetterBase
- **Purpose**: Base class for all Livepeer getter nodes
- **Responsibilities**:
  - Processing raw API results into ComfyUI-compatible formats
  - Tracking node instances and their associated jobs
  - Determining when a node should execute via IS_CHANGED
  - Managing job result caching and retrieval

## Data Flow

1. **Job Creation & Polling**:
   - User initiates a Livepeer operation through a requester node
   - LivepeerBase registers the job with LivepeerJobService
   - For async jobs, LivepeerBase starts a background thread for remote API polling
   - For sync jobs, the result is stored immediately in the service

2. **Job Status Check**:
   - Getter nodes track which job they're associated with
   - IS_CHANGED checks with the service if the job is complete
   - Returns dynamic value (time.time()) for pending jobs
   - Returns stable value (job_id) for completed jobs

3. **Result Processing**:
   - LivepeerBase updates the service when a job completes
   - When a getter node executes, it processes the raw result
   - Processed results are stored in the service
   - Subsequent executions retrieve cached results

4. **Cleanup**:
   - When nodes are deleted, they remove their association with jobs
   - The service automatically cleans up jobs that are no longer used

## Key Benefits

- **Clear Separation of Concerns**:
  - LivepeerBase handles API communication and polling
  - LivepeerJobService manages local state and lifecycle
  - LivepeerJobGetterBase focuses on processing results

- **Resource Efficiency**:
  - One polling thread per job regardless of how many getter nodes use it
  - Automatic cleanup of completed jobs

- **Simplified Node Code**:
  - Nodes focus on their specific processing logic
  - Common functionality abstracted into base classes

- **Robust Error Handling**:
  - Centralized error tracking and reporting
  - Clear state management between components

## Architecture Diagram

```
┌───────────────────┐                 ┌────────────────────┐
│                   │   Register Job  │                    │
│  LivepeerBase     ├────────────────►│  LivepeerJobService│
│  (Requester Nodes)│                 │    (Singleton)     │
│                   │                 │                    │
│  • API Polling    │  Update Status  │  • State Storage   │
│  • Job Execution  ├────────────────►│  • Node Tracking   │
└───────────────────┘                 │  • Cleanup         │
                                      │                    │
┌───────────────────┐  Query Status   │                    │
│                   │◄────────────────┤                    │
│ LivepeerJobGetter │                 │                    │
│  (Getter Nodes)   ├────────────────►│                    │
│                   │ Process Results │                    │
└───────────────────┘                 └────────────────────┘
``` 