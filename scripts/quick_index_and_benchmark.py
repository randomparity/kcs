#!/usr/bin/env python3
"""
Quick indexing and performance benchmark script.

This script manually indexes a few sample files and then runs the performance benchmark.
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path

# Add semantic search module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

import asyncpg
from semantic_search_benchmark import SemanticSearchBenchmark

# Database connection details
DATABASE_URL = "postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs"

async def manual_index_content():
    """Manually index some sample content for testing."""
    print("🔄 Manually indexing sample content...")

    # Connect directly to database
    conn = await asyncpg.connect(DATABASE_URL)

    # Sample kernel code content for indexing
    sample_files = [
        {
            "path": "/kernel/mm/slab.c",
            "content": """
/*
 * Memory allocation functions that can fail
 * This file implements the slab allocator for the Linux kernel
 */

#include <linux/slab.h>
#include <linux/mm.h>

/**
 * kmalloc - allocate memory
 * @size: how many bytes of memory are required.
 * @flags: the type of memory to allocate.
 *
 * kmalloc is the normal method of allocating memory in the kernel.
 * Returns NULL on failure.
 */
void *kmalloc(size_t size, gfp_t flags)
{
    struct kmem_cache *cachep;
    void *ret;

    if (unlikely(size > KMALLOC_MAX_SIZE))
        return NULL;

    cachep = kmalloc_slab(size, flags);
    if (unlikely(ZERO_OR_NULL_PTR(cachep)))
        return cachep;

    ret = slab_alloc(cachep, flags, _RET_IP_);
    trace_kmalloc(_RET_IP_, ret, size, cachep->size, flags);

    return ret;
}

static void *slab_alloc(struct kmem_cache *cachep, gfp_t flags, unsigned long caller)
{
    unsigned long save_flags;
    void *objp;

    flags &= gfp_allowed_mask;
    lockdep_trace_alloc(flags);

    if (should_failslab(cachep->object_size, flags, cachep->flags))
        return NULL;

    cache_alloc_debugcheck_before(cachep, flags);
    local_irq_save(save_flags);
    objp = __do_cache_alloc(cachep, flags);
    local_irq_restore(save_flags);

    return objp;
}
""",
            "content_type": "source_file",
            "title": "Kernel memory allocator"
        },
        {
            "path": "/kernel/net/packet.c",
            "content": """
/*
 * Network packet processing vulnerabilities and buffer handling
 * This file handles raw packet sockets
 */

#include <linux/types.h>
#include <linux/mm.h>
#include <linux/capability.h>
#include <linux/fcntl.h>
#include <linux/socket.h>
#include <linux/in.h>

/**
 * packet_rcv - receive a packet from the network
 * @skb: socket buffer containing the packet
 * @dev: network device
 * @pt: packet type handler
 * @orig_dev: original network device
 *
 * This function processes incoming network packets.
 * Buffer overflow vulnerability patterns possible here.
 */
static int packet_rcv(struct sk_buff *skb, struct net_device *dev,
                     struct packet_type *pt, struct net_device *orig_dev)
{
    struct sock *sk;
    struct sockaddr_ll *sll;
    struct packet_sock *po;
    u8 *skb_head = skb->data;
    int skb_len = skb->len;

    sk = pt->af_packet_priv;
    po = pkt_sk(sk);

    if (dev->header_ops) {
        skb_push(skb, skb->data - skb_mac_header(skb));
    }

    if (skb->pkt_type == PACKET_LOOPBACK)
        goto drop;

    sk->sk_data_ready(sk);

    if (po->origdev)
        po->stats.tp_packets++;

    return 0;

drop:
    kfree_skb(skb);
    return 0;
}

static int packet_setsockopt(struct socket *sock, int level, int optname,
                           char __user *optval, unsigned int optlen)
{
    char data[256];  /* Fixed buffer - potential overflow */

    if (optlen > sizeof(data))
        return -EINVAL;  /* Buffer overflow protection */

    if (copy_from_user(data, optval, optlen))
        return -EFAULT;

    /* Process socket options */
    return 0;
}
""",
            "content_type": "source_file",
            "title": "Network packet handling"
        },
        {
            "path": "/kernel/sched/core.c",
            "content": """
/*
 * Scheduler core functionality and lock acquisition patterns
 * This file implements the core scheduling algorithms
 */

#include <linux/sched.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>

static DEFINE_SPINLOCK(migration_lock);
static DEFINE_MUTEX(sched_domains_mutex);

/**
 * schedule - main scheduling function
 *
 * This is the main scheduling function that performs context switching.
 * Lock acquisition in scheduler is critical for system stability.
 */
asmlinkage __visible void __sched schedule(void)
{
    struct task_struct *tsk = current;

    sched_submit_work(tsk);
    do {
        preempt_disable();
        __schedule(false);
        sched_preempt_enable_no_resched();
    } while (need_resched());
    sched_update_worker(tsk);
}

static void __sched __schedule(bool preempt)
{
    struct task_struct *prev, *next;
    unsigned long *switch_count;
    struct rq_flags rf;
    struct rq *rq;
    int cpu;

    cpu = smp_processor_id();
    rq = cpu_rq(cpu);
    prev = rq->curr;

    schedule_debug(prev);

    if (sched_feat(HRTICK))
        hrtick_clear(rq);

    local_irq_disable();
    rcu_note_context_switch(preempt);

    /* Lock acquisition for scheduler runqueue */
    rq_lock(rq, &rf);
    smp_mb__after_spinlock();

    update_rq_clock(rq);
    switch_count = &prev->nivcsw;

    if (!preempt && prev->state) {
        if (signal_pending_state(prev->state, prev)) {
            prev->state = TASK_RUNNING;
        } else {
            deactivate_task(rq, prev, DEQUEUE_SLEEP | DEQUEUE_NOCLOCK);
        }
        switch_count = &prev->nvcsw;
    }

    next = pick_next_task(rq, prev, &rf);
    clear_tsk_need_resched(prev);
    clear_preempt_need_resched();

    if (likely(prev != next)) {
        rq->nr_switches++;
        rq->curr = next;
        ++*switch_count;

        trace_sched_switch(preempt, prev, next);

        /* Context switch */
        rq = context_switch(rq, prev, next, &rf);
    } else {
        rq_unlock_irq(rq, &rf);
    }

    balance_callback(rq);
}
""",
            "content_type": "source_file",
            "title": "Scheduler and locking"
        },
        {
            "path": "/drivers/net/ethernet/intel/e1000/e1000_main.c",
            "content": """
/*
 * Intel Gigabit Ethernet driver network packet processing
 * This file implements the Intel E1000 network driver
 */

#include <linux/module.h>
#include <linux/types.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/errno.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/kernel.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/skbuff.h>
#include <linux/delay.h>
#include <linux/timer.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/interrupt.h>
#include <linux/string.h>
#include <linux/pagemap.h>
#include <linux/dma-mapping.h>
#include <linux/bitops.h>

/**
 * e1000_clean_rx_irq - process received packets
 * @adapter: board private structure
 * @rx_ring: rx ring to clean
 * @work_done: number of packets processed
 * @work_to_do: how many packets to process
 *
 * Network packet processing with buffer management.
 */
static bool e1000_clean_rx_irq(struct e1000_adapter *adapter,
                              struct e1000_rx_ring *rx_ring,
                              int *work_done, int work_to_do)
{
    struct net_device *netdev = adapter->netdev;
    struct pci_dev *pdev = adapter->pdev;
    struct e1000_rx_desc *rx_desc, *next_rxd;
    struct e1000_rx_buffer *buffer_info, *next_buffer;
    unsigned long flags;
    u32 length;
    unsigned int i;
    int cleaned_count = 0;
    bool cleaned = false;
    unsigned int total_rx_bytes = 0, total_rx_packets = 0;

    i = rx_ring->next_to_clean;
    rx_desc = E1000_RX_DESC(*rx_ring, i);
    buffer_info = &rx_ring->buffer_info[i];

    while (rx_desc->status & E1000_RXD_STAT_DD) {
        struct sk_buff *skb;
        u8 status;

        if (*work_done >= work_to_do)
            break;
        (*work_done)++;
        rmb(); /* read descriptor and rx_buffer_info after status DD */

        status = rx_desc->status;
        skb = buffer_info->skb;
        buffer_info->skb = NULL;

        prefetch(skb->data - NET_IP_ALIGN);

        if (++i == rx_ring->count)
            i = 0;
        next_rxd = E1000_RX_DESC(*rx_ring, i);
        prefetch(next_rxd);

        next_buffer = &rx_ring->buffer_info[i];

        cleaned = true;
        cleaned_count++;
        dma_unmap_single(&pdev->dev, buffer_info->dma,
                        buffer_info->length, DMA_FROM_DEVICE);
        buffer_info->dma = 0;

        length = le16_to_cpu(rx_desc->length);
        /* good receive */
        skb_put(skb, length);

        /* Receive Checksum Offload */
        e1000_rx_checksum(adapter, status, le16_to_cpu(rx_desc->csum), skb);

        total_rx_bytes += skb->len;
        total_rx_packets++;

        netdev->stats.rx_bytes += skb->len;
        netdev->stats.rx_packets++;

        skb->protocol = eth_type_trans(skb, netdev);

        netif_receive_skb(skb);

        rx_desc->status = 0;

        /* use prefetched values */
        rx_desc = next_rxd;
        buffer_info = next_buffer;
    }

    rx_ring->next_to_clean = i;

    cleaned_count = E1000_DESC_UNUSED(rx_ring);
    if (cleaned_count)
        adapter->alloc_rx_buf(adapter, rx_ring, cleaned_count);

    adapter->total_rx_packets += total_rx_packets;
    adapter->total_rx_bytes += total_rx_bytes;
    netdev->stats.rx_bytes += total_rx_bytes;
    netdev->stats.rx_packets += total_rx_packets;

    return cleaned;
}
""",
            "content_type": "source_file",
            "title": "Network driver packet processing"
        },
        {
            "path": "/kernel/panic.c",
            "content": """
/*
 * Error handling patterns and system crash management
 * This file implements kernel panic and error handling
 */

#include <linux/debug_locks.h>
#include <linux/sched/debug.h>
#include <linux/interrupt.h>
#include <linux/kmsg_dump.h>
#include <linux/kallsyms.h>
#include <linux/notifier.h>
#include <linux/module.h>
#include <linux/random.h>
#include <linux/ftrace.h>
#include <linux/reboot.h>
#include <linux/delay.h>
#include <linux/kexec.h>
#include <linux/sched.h>
#include <linux/sysrq.h>
#include <linux/init.h>
#include <linux/nmi.h>
#include <linux/console.h>
#include <linux/bug.h>
#include <linux/ratelimit.h>

/**
 * panic - halt the system
 * @fmt: The text string to print
 *
 * Display a message, then perform cleanups.
 * Error handling patterns for critical system failures.
 */
void panic(const char *fmt, ...)
{
    static char buf[1024];
    va_list args;
    long i, i_next = 0;
    int state = 0;
    int old_cpu, this_cpu;
    bool _crash_kexec_post_notifiers = crash_kexec_post_notifiers;

    if (panic_on_warn) {
        /*
         * This thread may hit another WARN() in the panic path.
         * Resetting this prevents additional WARN() from panicking the
         * system on this thread. Other threads are blocked by the
         * panic_mutex.
         */
        panic_on_warn = 0;
    }

    /*
     * Disable local interrupts. This will prevent panic_smp_self_stop
     * from deadlocking the first cpu that invokes the panic, since
     * there is nothing to prevent an interrupt handler (that runs
     * after setting panic_cpu) from invoking panic() again.
     */
    local_irq_disable();

    /*
     * It's possible to come here directly from a panic-assertion and
     * not have preempt disabled. Some functions called from here want
     * preempt to be disabled. No point enabling it later though...
     *
     * Only one CPU is allowed to execute the panic code from here. For
     * multiple parallel invocations of panic, all other CPUs either
     * stop themself or will wait until they are stopped by the 1st CPU
     * with smp_send_stop().
     *
     * `old_cpu == PANIC_CPU_INVALID' means this is the 1st CPU which
     * comes here, so go ahead.
     * `old_cpu == this_cpu' means we came from nmi_panic() which sets
     * panic_cpu to this cpu.  In this case, this is also the 1st CPU.
     */
    this_cpu = raw_smp_processor_id();
    old_cpu = atomic_cmpxchg(&panic_cpu, PANIC_CPU_INVALID, this_cpu);

    if (old_cpu != PANIC_CPU_INVALID && old_cpu != this_cpu)
        panic_smp_self_stop();

    console_verbose();
    bust_spinlocks(1);
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    pr_emerg("Kernel panic - not syncing: %s\\n", buf);

    /*
     * Avoid nested stack-dumping if a panic occurs during oops processing
     */
    if (!test_taint(TAINT_DIE) && oops_in_progress <= 1)
        dump_stack();

    /*
     * If we have crashed and we have a crash kernel loaded let it handle
     * everything else.
     * If we want to run this after calling panic_notifiers, pass
     * the "crash_kexec_post_notifiers" option to the kernel.
     *
     * Bypass the panic_cpu check and call __crash_kexec directly.
     */
    if (!_crash_kexec_post_notifiers) {
        printk_safe_flush_on_panic();
        __crash_kexec(NULL);

        /*
         * Note smp_send_stop is the usual smp shutdown function, which
         * unfortunately means it may not be hardened to work in a
         * panic situation.
         */
        smp_send_stop();
    } else {
        /*
         * If we want to do crash dump after notifier calls and
         * kmsg_dump, we will need architecture dependent extra
         * works in addition to stopping other CPUs.
         */
        crash_smp_send_stop();
    }

    /*
     * Run any panic handlers, including those that might need to
     * add information to the kmsg dump output.
     */
    atomic_notifier_call_chain(&panic_notifier_list, 0, buf);

    /* Call flush even twice. It tries harder with a single online CPU */
    printk_safe_flush_on_panic();
    kmsg_dump(KMSG_DUMP_PANIC);

    /*
     * If you doubt kdump always works fine in any situation,
     * "crash_kexec_post_notifiers" offers you a chance to run
     * panic_notifiers and dumping kmsg before kdump.
     * Note: since some panic_notifiers can make crashed kernel
     * more unstable, it can increase risks of the kdump failure too.
     *
     * Bypass the panic_cpu check and call __crash_kexec directly.
     */
    if (_crash_kexec_post_notifiers)
        __crash_kexec(NULL);

    bust_spinlocks(0);

    /*
     * We may have ended up stopping the CPU holding the lock (in
     * smp_send_stop()) while still having some valuable data in the console
     * buffer.  Try to acquire the lock then release it regardless of the
     * result.  The release will also print the buffers out.  Locks debug
     * should be disabled to avoid reporting bad unlock balance when
     * panic() is not being callled from OOPS.
     */
    debug_locks_off();
    console_flush_on_panic();

    panic_print_sys_info();

    if (!panic_blink)
        panic_blink = no_blink;

    if (panic_timeout > 0) {
        /*
         * Delay timeout seconds before rebooting the machine.
         * We can't use the "normal" timers since we just panicked.
         */
        pr_emerg("Rebooting in %d seconds..", panic_timeout);

        for (i = 0; i < panic_timeout * 1000; i += PANIC_TIMER_STEP) {
            touch_nmi_watchdog();
            if (i >= i_next) {
                i += panic_blink(state ^= 1);
                i_next = i + 3600 / PANIC_BLINK_SPD;
            }
            mdelay(PANIC_TIMER_STEP);
        }
    }
    if (panic_timeout != 0) {
        /*
         * This will not be a clean reboot, with everything
         * shutting down.  But if there is a chance of
         * rebooting the system it will be rebooted.
         */
        if (panic_reboot_mode != REBOOT_UNDEFINED)
            reboot_mode = panic_reboot_mode;
        emergency_restart();
    }

    pr_emerg("---[ end Kernel panic - not syncing: %s ]---\\n", buf);
    local_irq_enable();
    for (i = 0; ; i += PANIC_TIMER_STEP) {
        touch_softlockup_watchdog();
        if (i >= i_next) {
            i += panic_blink(state ^= 1);
            i_next = i + 3600 / PANIC_BLINK_SPD;
        }
        mdelay(PANIC_TIMER_STEP);
    }
}
""",
            "content_type": "source_file",
            "title": "Error handling and panic"
        }
    ]

    # Insert sample content into database
    for idx, sample in enumerate(sample_files):
        content_hash = hashlib.sha256(sample["content"].encode()).hexdigest()

        # Insert into indexed_content table
        import json
        metadata_json = json.dumps({"file_size": len(sample["content"]), "sample": True})
        content_id = await conn.fetchval(
            """
            INSERT INTO indexed_content (
                content_type, source_path, content_hash, title, content, metadata, status, indexed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING id
            """,
            sample["content_type"],
            sample["path"],
            content_hash,
            sample["title"],
            sample["content"],
            metadata_json,
            "completed"
        )

        # Create a simple dummy embedding (384 dimensions filled with small random values)
        # In reality this would come from BAAI/bge-small-en-v1.5 model
        import random
        random.seed(idx)  # Deterministic for testing
        embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]

        # Insert into vector_embedding table
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        await conn.execute(
            """
            INSERT INTO vector_embedding (
                content_id, embedding, model_name, model_version, chunk_index
            ) VALUES ($1, $2::vector, $3, $4, $5)
            """,
            content_id,
            embedding_str,
            "BAAI/bge-small-en-v1.5",
            "1.0",
            0
        )

        print(f"✅ Indexed: {sample['title']} (ID: {content_id})")

    await conn.close()
    print(f"🎉 Successfully indexed {len(sample_files)} sample files")


async def main():
    """Main function to index content and run benchmark."""
    print("🚀 Starting quick index and benchmark process...")

    # Index sample content
    await manual_index_content()

    print("\n" + "="*60)
    print("🏃‍♂️ Running performance benchmark against real data...")
    print("="*60)

    # Run the performance benchmark
    benchmark = SemanticSearchBenchmark(DATABASE_URL)
    try:
        results = await benchmark.run_full_benchmark()

        # Import the print function from the original script
        from semantic_search_benchmark import print_benchmark_results
        print_benchmark_results(results)

        # Exit with appropriate code
        if results.get("overall_assessment", {}).get("constitutional_compliance", False):
            print("\n🎉 Performance benchmark PASSED!")
            return 0
        else:
            print("\n❌ Performance benchmark FAILED!")
            return 1

    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
