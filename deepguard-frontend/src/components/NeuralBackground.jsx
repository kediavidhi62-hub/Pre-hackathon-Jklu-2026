import { useEffect, useRef } from 'react'

/**
 * Full-screen animated neural network mesh background.
 * - Hundreds of pulsing/flickering nodes in blue & purple
 * - Dynamic polygon connections
 * - Scanning light wave
 * - Floating micro-particles
 * - Parallax depth layers
 * - Smooth, performant 60fps loop
 */
export default function NeuralBackground() {
    const canvasRef = useRef(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        let animId
        let w, h, dpr
        let mouse = { x: -9999, y: -9999 }

        /* ── Config ── */
        const NODE_COUNT = 120
        const PARTICLE_COUNT = 60
        const CONNECTION_DIST = 140
        const MOUSE_DIST = 200

        /* ── Colors (brand blue #2563EB, brand purple #7C3AED) ── */
        const BLUE = { r: 37, g: 99, b: 235 }
        const PURPLE = { r: 124, g: 58, b: 237 }

        /* ── Resize ── */
        const resize = () => {
            dpr = Math.min(window.devicePixelRatio || 1, 2)
            w = window.innerWidth
            h = window.innerHeight
            canvas.width = w * dpr
            canvas.height = h * dpr
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        }

        /* ── Node class ── */
        class Node {
            constructor(layer) {
                this.layer = layer // 0 = back, 1 = mid, 2 = front
                const speed = 0.15 + layer * 0.1
                this.x = Math.random() * w
                this.y = Math.random() * h
                this.vx = (Math.random() - 0.5) * speed
                this.vy = (Math.random() - 0.5) * speed
                this.baseRadius = 1.2 + layer * 0.8 + Math.random() * 1.2
                this.radius = this.baseRadius
                const c = Math.random() > 0.45 ? BLUE : PURPLE
                this.r = c.r; this.g = c.g; this.b = c.b
                this.baseAlpha = 0.3 + layer * 0.15 + Math.random() * 0.2
                this.alpha = this.baseAlpha
                this.pulseSpeed = 0.015 + Math.random() * 0.025
                this.pulsePhase = Math.random() * Math.PI * 2
                this.flickerTimer = Math.random() * 500
                this.flickerInterval = 200 + Math.random() * 600
                this.isFlickering = false
            }
            update(t) {
                // Drift
                this.x += this.vx
                this.y += this.vy

                // Parallax mouse influence
                if (mouse.x > -9000) {
                    const factor = (this.layer + 1) * 0.0003
                    this.x += (mouse.x - w / 2) * factor
                    this.y += (mouse.y - h / 2) * factor
                }

                // Wrap
                if (this.x < -20) this.x = w + 20
                if (this.x > w + 20) this.x = -20
                if (this.y < -20) this.y = h + 20
                if (this.y > h + 20) this.y = -20

                // Pulse
                const pulse = Math.sin(t * this.pulseSpeed + this.pulsePhase)
                this.radius = this.baseRadius + pulse * 0.6
                this.alpha = this.baseAlpha + pulse * 0.12

                // Flicker
                this.flickerTimer++
                if (this.flickerTimer > this.flickerInterval) {
                    this.isFlickering = true
                    if (this.flickerTimer > this.flickerInterval + 8) {
                        this.isFlickering = false
                        this.flickerTimer = 0
                        this.flickerInterval = 200 + Math.random() * 600
                    }
                }
                if (this.isFlickering) {
                    this.alpha = Math.random() > 0.5 ? this.baseAlpha * 1.8 : this.baseAlpha * 0.3
                }
            }
            draw() {
                // Glow halo
                const grd = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 5)
                grd.addColorStop(0, `rgba(${this.r},${this.g},${this.b},${this.alpha * 0.5})`)
                grd.addColorStop(0.4, `rgba(${this.r},${this.g},${this.b},${this.alpha * 0.12})`)
                grd.addColorStop(1, `rgba(${this.r},${this.g},${this.b},0)`)
                ctx.beginPath()
                ctx.fillStyle = grd
                ctx.arc(this.x, this.y, this.radius * 5, 0, Math.PI * 2)
                ctx.fill()

                // Core
                ctx.beginPath()
                ctx.fillStyle = `rgba(${this.r},${this.g},${this.b},${Math.min(this.alpha + 0.35, 1)})`
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
                ctx.fill()
            }
        }

        /* ── Micro-particle class ── */
        class MicroParticle {
            constructor() {
                this.x = Math.random() * w
                this.y = Math.random() * h
                this.vx = (Math.random() - 0.5) * 0.3
                this.vy = -0.2 - Math.random() * 0.3 // drift upward
                this.radius = 0.5 + Math.random() * 0.8
                this.alpha = 0.15 + Math.random() * 0.2
                this.life = Math.random()
            }
            update() {
                this.x += this.vx
                this.y += this.vy
                this.life += 0.002
                if (this.life > 1) {
                    this.x = Math.random() * w
                    this.y = h + 10
                    this.life = 0
                }
                this.alpha = 0.15 + Math.sin(this.life * Math.PI) * 0.2
            }
            draw() {
                ctx.beginPath()
                ctx.fillStyle = `rgba(37, 99, 235, ${this.alpha})`
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
                ctx.fill()
            }
        }

        /* ── Init ── */
        let nodes = []
        let microParticles = []

        const init = () => {
            nodes = []
            for (let i = 0; i < NODE_COUNT; i++) {
                const layer = i < NODE_COUNT * 0.3 ? 0 : i < NODE_COUNT * 0.7 ? 1 : 2
                nodes.push(new Node(layer))
            }
            microParticles = Array.from({ length: PARTICLE_COUNT }, () => new MicroParticle())
        }

        /* ── Draw connections ── */
        const drawConnections = (scanX) => {
            for (let i = 0; i < nodes.length; i++) {
                const a = nodes[i]
                for (let j = i + 1; j < nodes.length; j++) {
                    const b = nodes[j]
                    if (Math.abs(a.layer - b.layer) > 1) continue // only connect same/adjacent layers
                    const dx = a.x - b.x
                    const dy = a.y - b.y
                    const dist = Math.sqrt(dx * dx + dy * dy)

                    if (dist < CONNECTION_DIST) {
                        let opacity = (1 - dist / CONNECTION_DIST) * 0.2

                        // Brighten near scan wave
                        const midX = (a.x + b.x) / 2
                        const scanDist = Math.abs(midX - scanX)
                        if (scanDist < 120) {
                            opacity += (1 - scanDist / 120) * 0.15
                        }

                        ctx.beginPath()
                        ctx.strokeStyle = `rgba(37, 99, 235, ${opacity})`
                        ctx.lineWidth = 0.6 + (a.layer * 0.15)
                        ctx.moveTo(a.x, a.y)
                        ctx.lineTo(b.x, b.y)
                        ctx.stroke()
                    }
                }

                // Mouse connections
                const mdx = a.x - mouse.x
                const mdy = a.y - mouse.y
                const mDist = Math.sqrt(mdx * mdx + mdy * mdy)
                if (mDist < MOUSE_DIST) {
                    const opacity = (1 - mDist / MOUSE_DIST) * 0.45
                    ctx.beginPath()
                    ctx.strokeStyle = `rgba(124, 58, 237, ${opacity})`
                    ctx.lineWidth = 1.2
                    ctx.moveTo(a.x, a.y)
                    ctx.lineTo(mouse.x, mouse.y)
                    ctx.stroke()
                }
            }
        }

        /* ── Scanning wave ── */
        let scanX = -200

        const drawScanWave = () => {
            const gradient = ctx.createLinearGradient(scanX - 100, 0, scanX + 100, 0)
            gradient.addColorStop(0, 'rgba(37, 99, 235, 0)')
            gradient.addColorStop(0.5, 'rgba(124, 58, 237, 0.04)')
            gradient.addColorStop(1, 'rgba(37, 99, 235, 0)')
            ctx.fillStyle = gradient
            ctx.fillRect(scanX - 100, 0, 200, h)
        }

        /* ── Animate ── */
        let t = 0
        const animate = () => {
            ctx.clearRect(0, 0, w, h)
            t++

            // Scan wave movement
            scanX += 1.5
            if (scanX > w + 200) scanX = -200

            // Update & draw in layer order (back → front)
            const sorted = [...nodes].sort((a, b) => a.layer - b.layer)

            sorted.forEach((n) => { n.update(t); })
            drawConnections(scanX)
            drawScanWave()
            sorted.forEach((n) => n.draw())

            // Micro particles
            microParticles.forEach((p) => { p.update(); p.draw() })

            animId = requestAnimationFrame(animate)
        }

        /* ── Events ── */
        const onMouse = (e) => { mouse.x = e.clientX; mouse.y = e.clientY }
        const onLeave = () => { mouse.x = -9999; mouse.y = -9999 }
        const onResize = () => { resize(); init() }

        resize()
        init()
        animate()

        window.addEventListener('resize', onResize)
        window.addEventListener('mousemove', onMouse)
        window.addEventListener('mouseleave', onLeave)

        return () => {
            cancelAnimationFrame(animId)
            window.removeEventListener('resize', onResize)
            window.removeEventListener('mousemove', onMouse)
            window.removeEventListener('mouseleave', onLeave)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
        />
    )
}
