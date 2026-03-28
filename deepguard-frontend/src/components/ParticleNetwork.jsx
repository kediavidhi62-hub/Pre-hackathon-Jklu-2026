import { useEffect, useRef } from 'react'

export default function ParticleNetwork() {
    const canvasRef = useRef(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        let animationId
        let particles = []
        const PARTICLE_COUNT = 90
        const CONNECTION_DIST = 160
        const MOUSE_DIST = 220

        let mouse = { x: -1000, y: -1000 }
        let w, h

        const resize = () => {
            const dpr = window.devicePixelRatio || 1
            w = canvas.offsetWidth
            h = canvas.offsetHeight
            canvas.width = w * dpr
            canvas.height = h * dpr
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        }

        class Particle {
            constructor() {
                this.x = Math.random() * (w || 800)
                this.y = Math.random() * (h || 600)
                this.vx = (Math.random() - 0.5) * 0.5
                this.vy = (Math.random() - 0.5) * 0.5
                this.radius = Math.random() * 2.5 + 1.5
                // Blue (#2563EB) or Purple (#7C3AED)
                const isBlue = Math.random() > 0.4
                this.r = isBlue ? 37 : 124
                this.g = isBlue ? 99 : 58
                this.b = isBlue ? 235 : 237
                this.baseAlpha = Math.random() * 0.4 + 0.4
                this.pulseSpeed = Math.random() * 0.03 + 0.015
                this.pulsePhase = Math.random() * Math.PI * 2
            }
            update(t) {
                this.x += this.vx
                this.y += this.vy
                if (this.x < 0 || this.x > w) this.vx *= -1
                if (this.y < 0 || this.y > h) this.vy *= -1
                this.alpha = this.baseAlpha + Math.sin(t * this.pulseSpeed + this.pulsePhase) * 0.2
            }
            draw() {
                // Outer glow
                const grd = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 6)
                grd.addColorStop(0, `rgba(${this.r},${this.g},${this.b},${this.alpha * 0.6})`)
                grd.addColorStop(0.5, `rgba(${this.r},${this.g},${this.b},${this.alpha * 0.15})`)
                grd.addColorStop(1, `rgba(${this.r},${this.g},${this.b},0)`)
                ctx.beginPath()
                ctx.fillStyle = grd
                ctx.arc(this.x, this.y, this.radius * 6, 0, Math.PI * 2)
                ctx.fill()

                // Core
                ctx.beginPath()
                ctx.fillStyle = `rgba(${this.r},${this.g},${this.b},${Math.min(this.alpha + 0.3, 1)})`
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
                ctx.fill()
            }
        }

        const init = () => {
            particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle())
        }

        const drawConnections = () => {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x
                    const dy = particles[i].y - particles[j].y
                    const dist = Math.sqrt(dx * dx + dy * dy)

                    if (dist < CONNECTION_DIST) {
                        const opacity = (1 - dist / CONNECTION_DIST) * 0.35
                        ctx.beginPath()
                        ctx.strokeStyle = `rgba(37, 99, 235, ${opacity})`
                        ctx.lineWidth = 1
                        ctx.moveTo(particles[i].x, particles[i].y)
                        ctx.lineTo(particles[j].x, particles[j].y)
                        ctx.stroke()
                    }
                }

                // Mouse hover connections
                const mdx = particles[i].x - mouse.x
                const mdy = particles[i].y - mouse.y
                const mDist = Math.sqrt(mdx * mdx + mdy * mdy)
                if (mDist < MOUSE_DIST) {
                    const opacity = (1 - mDist / MOUSE_DIST) * 0.6
                    ctx.beginPath()
                    ctx.strokeStyle = `rgba(124, 58, 237, ${opacity})`
                    ctx.lineWidth = 1.5
                    ctx.moveTo(particles[i].x, particles[i].y)
                    ctx.lineTo(mouse.x, mouse.y)
                    ctx.stroke()
                }
            }
        }

        let t = 0
        const animate = () => {
            ctx.clearRect(0, 0, w, h)
            t++
            particles.forEach((p) => { p.update(t); p.draw() })
            drawConnections()
            animationId = requestAnimationFrame(animate)
        }

        const handleMouseMove = (e) => {
            const rect = canvas.getBoundingClientRect()
            mouse.x = e.clientX - rect.left
            mouse.y = e.clientY - rect.top
        }
        const handleMouseLeave = () => { mouse.x = -1000; mouse.y = -1000 }

        resize()
        init()
        animate()

        const handleResize = () => { resize(); init() }
        window.addEventListener('resize', handleResize)
        canvas.addEventListener('mousemove', handleMouseMove)
        canvas.addEventListener('mouseleave', handleMouseLeave)

        return () => {
            cancelAnimationFrame(animationId)
            window.removeEventListener('resize', handleResize)
            canvas.removeEventListener('mousemove', handleMouseMove)
            canvas.removeEventListener('mouseleave', handleMouseLeave)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
        />
    )
}
