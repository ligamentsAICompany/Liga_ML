# Hardware Support Fine-Tuning Dataset

## Example 1
User: My gaming PC shuts down after ten minutes of playing, and the GPU hotspot reaches 105 C.
Assistant: Stop gaming until temperatures are controlled. Check that the GPU fans spin, remove dust from the heatsink, improve case airflow, and verify the power connectors are fully seated. If the hotspot remains above the safe range after cleaning and airflow fixes, contact the GPU vendor because repasting or repair may affect warranty.
Category: GPU overheating
Safety: Power down before opening the case, avoid touching fan blades while powered, and preserve warranty stickers.

## Example 2
User: My CPU clock drops from 4.6 GHz to 1.2 GHz during video exports.
Assistant: This looks like thermal or power throttling. Monitor CPU package temperature, VRM temperature, and power limits during the export. Clean the cooler, confirm the pump or fan curve works, re-seat the cooler if temperatures spike instantly, and check BIOS power limit settings.
Category: CPU throttling
Safety: Shut down and unplug the system before re-seating the cooler.

## Example 3
User: I bought DDR5-6000 RAM, but my motherboard only boots at 4800 MT/s.
Assistant: First confirm the motherboard QVL and CPU memory controller support the kit speed. Update BIOS, install the modules in the recommended slots, then enable EXPO or XMP. If it fails, use a lower stable profile such as 5600 MT/s and run a memory test.
Category: RAM compatibility
Safety: Disconnect power before moving DIMMs and avoid forcing modules into the slot.

## Example 4
User: My SSD disappears from BIOS after the computer sleeps.
Assistant: Update the SSD firmware and motherboard BIOS, disable aggressive PCIe link power management as a test, and check whether the M.2 drive is overheating. Re-seat the drive and verify the standoff is installed correctly. Back up important data immediately because disappearing drives can indicate failure.
Category: SSD failure
Safety: Back up data before firmware changes and unplug power before re-seating the SSD.

## Example 5
User: The motherboard shows a DRAM debug LED and never reaches POST.
Assistant: Power off, clear CMOS, then try one memory module in the primary recommended slot. Re-seat the RAM until both latches click, inspect for bent CPU socket pins if applicable, and test with known-good memory if available.
Category: Motherboard POST errors
Safety: Remove power and discharge the system before handling RAM or CPU components.

## Example 6
User: My new graphics card turns on, but the system reboots when a game starts.
Assistant: Check the PSU wattage and quality against the GPU recommendation, confirm separate PCIe power cables are used instead of one daisy-chain cable, and inspect event logs for power loss. If the issue appears only under GPU load, test with a known-good PSU.
Category: Power supply
Safety: Do not open a PSU; capacitors can retain dangerous charge.

## Example 7
User: My laptop battery drops from 40 percent to 5 percent in a few minutes.
Assistant: Generate a battery health report, check cycle count and full charge capacity, and recalibrate only if the health looks normal. If capacity is degraded or the laptop shuts off suddenly, replace the battery through an authorized service channel.
Category: Laptop battery
Safety: Stop using swollen batteries immediately and do not puncture or compress them.

## Example 8
User: Should I update BIOS before installing a newer CPU?
Assistant: Check the motherboard CPU support list and required BIOS version. If your current BIOS is older than required, update it before swapping CPUs, using the vendor instructions and a stable power source. Reset settings after the update if boot issues occur.
Category: BIOS/UEFI guidance
Safety: Do not interrupt a BIOS update; use AC power or a UPS when possible.

## Example 9
User: My PC fans ramp up and down every few seconds at idle.
Assistant: Review fan curves in BIOS or vendor software and add a smoother temperature response delay. Clean dust filters, confirm fan headers report stable RPM, and check whether a background process is causing short CPU temperature spikes.
Category: Fan noise
Safety: Keep fingers and cables away from spinning fans.

## Example 10
User: I see green artifacts on the screen after the GPU driver loads.
Assistant: Try a different cable and monitor port, perform a clean driver install, and test the GPU under light and heavy loads. If artifacts appear in BIOS or on another system, the GPU memory or display output may be failing.
Category: Display artifacts
Safety: Power off before swapping cables inside the case.

## Example 11
User: My USB keyboard works in Windows but not in BIOS.
Assistant: Try rear motherboard USB ports, enable legacy USB support if available, and update BIOS. If the keyboard uses a hub or wireless dongle, connect it directly and test a simple wired keyboard.
Category: Peripherals
Safety: Avoid repeatedly hot-plugging damaged USB devices.

## Example 12
User: Do I need an anti-static wrist strap to replace RAM?
Assistant: A wrist strap is helpful, but at minimum work on a non-carpeted surface, unplug the PC, touch grounded metal before handling parts, and hold RAM by the edges. Store modules in anti-static bags when not installed.
Category: ESD handling
Safety: Never work inside a powered computer.

## Example 13
User: My prebuilt PC is under warranty, but I want to replace the cooler.
Assistant: Check the warranty terms before modifying hardware. Some vendors allow RAM or storage upgrades but restrict cooler changes. If cooling is inadequate, document temperatures and contact support before replacing parts.
Category: Warranty
Safety: Avoid modifications that could void coverage unless you accept that risk.

## Example 14
User: My hard drive is clicking and Windows asks to repair it.
Assistant: Stop using the drive for normal work, copy critical files to another disk, and check SMART health from a trusted tool. Do not run repeated repair attempts before backup because failing drives can degrade further.
Category: Backup before repair
Safety: Prioritize backup before diagnostics that stress the drive.

## Example 15
User: My GPU memory temperature is 110 C while mining, but core temperature is 70 C.
Assistant: Memory junction temperature can overheat even when the core looks fine. Reduce power limit and memory clocks, improve airflow over the card backplate, and stop the workload if temperatures remain excessive. Persistent high memory temperatures may require professional service.
Category: GPU overheating
Safety: Avoid disassembly if it would void warranty or expose you to damage risk.

## Example 16
User: My CPU hits 100 C immediately after boot.
Assistant: Immediate overheating often means poor cooler contact, missing thermal paste, a failed pump, or an unplugged CPU fan. Shut down, inspect mounting pressure, verify the fan or pump header, and reapply thermal paste if needed.
Category: CPU throttling
Safety: Let the system cool and unplug it before touching the cooler.

## Example 17
User: I mixed two RAM kits with the same speed, and now games crash.
Assistant: Mixed kits can be unstable even with matching labels. Disable XMP or EXPO, test each kit separately, then test together at a lower speed and looser timings. For reliability, use a matched kit validated for your board.
Category: RAM compatibility
Safety: Power off before removing or installing memory.

## Example 18
User: My NVMe SSD is slow after copying large files for several minutes.
Assistant: Check drive temperature and whether the workload exceeds the SLC cache. Install a proper heatsink if the drive throttles, update firmware, and keep free space available. Compare results with the manufacturer benchmark expectations.
Category: SSD failure
Safety: Back up important data before firmware updates.

## Example 19
User: The PC turns on for one second, then turns off repeatedly.
Assistant: Disconnect nonessential devices, check CPU power and 24-pin motherboard power cables, clear CMOS, and test the PSU. Also inspect for shorts from misplaced standoffs or loose screws under the motherboard.
Category: Motherboard POST errors
Safety: Do not keep power cycling if you smell burning or see sparks.

## Example 20
User: Can a weak PSU cause blue screens instead of just shutdowns?
Assistant: Yes. Voltage instability can cause blue screens, driver resets, or data errors under load. Check event logs, test CPU and GPU loads separately, and verify the PSU has sufficient capacity and proper cabling.
Category: Power supply
Safety: Replace suspect PSUs rather than attempting internal repair.

## Example 21
User: My laptop only charges when the cable is held at an angle.
Assistant: The adapter plug, cable, or charging port may be worn. Test a known-good charger with the correct rating, inspect the port for debris, and seek service if the port is loose. Avoid using it in a strained position because arcing can damage the board.
Category: Laptop battery
Safety: Stop using chargers with exposed wires or overheating bricks.

## Example 22
User: After enabling secure boot, my PC no longer boots from the old SSD.
Assistant: The old installation may be using legacy boot or an unsigned bootloader. Re-enter UEFI, check boot mode, and disable secure boot temporarily if you need to recover files. For a permanent fix, convert the installation to UEFI/GPT or reinstall cleanly.
Category: BIOS/UEFI guidance
Safety: Back up files before changing partitions or boot mode.

## Example 23
User: My case fan makes a grinding noise at low speed.
Assistant: A grinding fan usually has worn bearings or obstruction. Check that no cable touches the blades, clean dust, and test the fan at different speeds. Replace the fan if the noise persists.
Category: Fan noise
Safety: Power off before moving cables near fans.

## Example 24
User: The monitor shows random black flashes with a new high-refresh cable.
Assistant: Verify the cable is certified for the resolution and refresh rate, try another port, and lower refresh rate as a test. Update GPU drivers and monitor firmware if available. If flashes stop with a different cable, replace the cable.
Category: Display artifacts
Safety: Avoid tugging cables while devices are powered if ports feel loose.

## Example 25
User: My external drive disconnects when I plug it into the front USB port.
Assistant: Front-panel USB may provide less stable power or have a loose header. Try rear motherboard ports, a powered hub, and a different cable. Inspect the front-panel connector on the motherboard if rear ports work reliably.
Category: Peripherals
Safety: Unplug the PC before checking internal front-panel headers.

## Example 26
User: I touched the GPU contacts by accident. Is it ruined?
Assistant: It is probably fine if there was no static discharge or contamination. Power off, handle the card by the edges, and gently clean contacts only if visibly dirty using high-purity isopropyl alcohol and a lint-free swab. Let it dry completely before installation.
Category: ESD handling
Safety: Do not use water or abrasive materials on contacts.

## Example 27
User: The manufacturer says opening the laptop voids warranty, but the SSD is failing.
Assistant: If warranty terms restrict opening the laptop, document the SSD symptoms and start a warranty claim first. Ask whether they support user data backup or drive replacement. If data is critical, back it up before shipping when possible.
Category: Warranty
Safety: Do not delay backup if the drive is still readable.

## Example 28
User: Windows repair wants to reset my PC after a failed update.
Assistant: Before resetting, back up personal files from recovery mode or by removing the drive and connecting it to another computer. Then try system restore or uninstalling the latest update. Use reset only after data is safe.
Category: Backup before repair
Safety: Confirm backups open correctly before destructive recovery.

## Example 29
User: My GPU fans stay at zero RPM until the card is already hot.
Assistant: Many GPUs use zero-RPM mode at idle, but fans should start under load. Check the fan curve, update GPU control software, and confirm the fans spin with a manual test. If fans do not respond under load, stop stressing the GPU and request service.
Category: GPU overheating
Safety: Do not run high-load tests if fans are not spinning.

## Example 30
User: My CPU score is much lower after I changed the cooler.
Assistant: Check mounting pressure, thermal paste spread, CPU fan or pump speed, and whether the cooler protective film was removed. Monitor temperatures and power limits during the benchmark to identify thermal throttling.
Category: CPU throttling
Safety: Shut down before loosening cooler screws.

## Example 31
User: My motherboard supports 128 GB RAM, but four sticks will not run at rated XMP.
Assistant: Four DIMMs stress the memory controller more than two. Update BIOS, test each module, then lower memory speed or loosen timings. Use a memory kit sold as a four-DIMM set for best stability.
Category: RAM compatibility
Safety: Avoid excessive DRAM voltage beyond vendor guidance.

## Example 32
User: My SATA SSD shows many reallocated sectors.
Assistant: Reallocated sectors indicate the SSD has replaced failing blocks. Back up data immediately, check firmware and SMART health, and plan replacement. Do not rely on the drive for important data.
Category: SSD failure
Safety: Backup comes before stress testing or secure erase.

## Example 33
User: The motherboard VGA light stays on, but my CPU has no integrated graphics.
Assistant: Make sure the monitor cable is connected to the graphics card, not the motherboard. Re-seat the GPU, check PCIe power cables, try another PCIe slot if available, and clear CMOS. Test with another GPU if possible.
Category: Motherboard POST errors
Safety: Power off and unplug before re-seating the GPU.

## Example 34
User: My PC smells like hot plastic near the power cable.
Assistant: Shut it down immediately and unplug it. Inspect the power cable, PSU socket, and wall outlet for heat damage. Replace damaged cables and have the PSU or outlet inspected; do not continue using a system with burning smells.
Category: Power supply
Safety: Fire risk; stop use until the cause is found.

## Example 35
User: My laptop battery is swollen and the trackpad is lifting.
Assistant: Stop using and charging the laptop. Power it down, place it on a nonflammable surface, and arrange professional battery replacement. Do not press the trackpad back down or puncture the battery.
Category: Laptop battery
Safety: Swollen lithium batteries are hazardous and need careful handling.

## Example 36
User: I changed a BIOS setting and now the PC will not display anything.
Assistant: Clear CMOS using the motherboard jumper or button according to the manual. If needed, remove the CMOS battery briefly with power disconnected. Then boot with default settings and change one setting at a time.
Category: BIOS/UEFI guidance
Safety: Unplug power before using CMOS jumpers or removing the battery.

## Example 37
User: My AIO cooler makes gurgling noises after I moved the PC.
Assistant: Some trapped air noise can occur after movement. Mount the radiator so the pump is not the highest point, run the pump at a steady speed briefly, and monitor CPU temperature. Replace the cooler if noise persists with poor temperatures.
Category: Fan noise
Safety: Check for leaks and power off if liquid is visible.

## Example 38
User: My screen has checkerboard artifacts only when VRAM usage is high.
Assistant: That pattern can indicate GPU memory instability. Remove overclocks, update drivers, improve cooling, and test with a VRAM stress tool. If artifacts continue at stock settings, the GPU may need warranty service.
Category: Display artifacts
Safety: Stop tests if artifacts worsen or the system crashes repeatedly.

## Example 39
User: My Bluetooth mouse lags when a USB 3 drive is connected.
Assistant: USB 3 devices can interfere with 2.4 GHz wireless receivers. Move the Bluetooth or wireless dongle away using an extension cable, try rear ports, update chipset drivers, and keep the drive cable away from the receiver.
Category: Peripherals
Safety: Use intact cables and avoid sharply bending USB connectors.

## Example 40
User: I am replacing a motherboard. What should I do before the repair?
Assistant: Back up important files, record BitLocker or recovery keys, note cable locations with photos, and prepare drivers for the new board. After replacement, check boot mode, fan headers, and Windows activation.
Category: Backup before repair
Safety: Disconnect power and use ESD precautions throughout the repair.
