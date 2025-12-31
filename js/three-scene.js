/**
 * Three.js 3D Background Scene
 * Floating mathematical symbols and particle effects
 */

class QuantScene {
    constructor() {
        this.container = document.getElementById('three-canvas');
        this.width = window.innerWidth;
        this.height = window.innerHeight;
        this.mouseX = 0;
        this.mouseY = 0;

        this.init();
        this.createParticles();
        this.createFloatingSymbols();
        this.createGrid();
        this.animate();
        this.bindEvents();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.0008);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.width / this.height,
            1,
            2000
        );
        this.camera.position.z = 800;
        this.camera.position.y = 100;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.container,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x0a0a0f, 1);

        // Clock for animations
        this.clock = new THREE.Clock();
    }

    createParticles() {
        const particleCount = 2000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            positions[i3] = (Math.random() - 0.5) * 2000;
            positions[i3 + 1] = (Math.random() - 0.5) * 2000;
            positions[i3 + 2] = (Math.random() - 0.5) * 2000;
            sizes[i] = Math.random() * 2 + 0.5;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        // Custom shader material for particles
        const material = new THREE.PointsMaterial({
            color: 0x00d4ff,
            size: 2,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending,
            sizeAttenuation: true
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    createFloatingSymbols() {
        this.symbols = [];
        const symbolChars = ['∫', 'Σ', '√', 'σ', 'κ', 'θ', 'ρ', 'φ', '∂', 'Δ', 'π', 'μ', 'λ'];

        // Create canvas-based text sprites
        symbolChars.forEach((char, index) => {
            const sprite = this.createTextSprite(char);

            // Random position
            sprite.position.x = (Math.random() - 0.5) * 1500;
            sprite.position.y = (Math.random() - 0.5) * 1000;
            sprite.position.z = (Math.random() - 0.5) * 1000 - 200;

            // Store animation properties
            sprite.userData = {
                originalY: sprite.position.y,
                speed: 0.5 + Math.random() * 0.5,
                rotationSpeed: (Math.random() - 0.5) * 0.02,
                phase: Math.random() * Math.PI * 2
            };

            this.symbols.push(sprite);
            this.scene.add(sprite);
        });
    }

    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        const size = 256;
        canvas.width = size;
        canvas.height = size;

        const context = canvas.getContext('2d');

        // Background (transparent)
        context.fillStyle = 'transparent';
        context.fillRect(0, 0, size, size);

        // Text
        context.font = 'bold 180px JetBrains Mono, monospace';
        context.textAlign = 'center';
        context.textBaseline = 'middle';

        // Glow effect
        context.shadowColor = '#00d4ff';
        context.shadowBlur = 20;
        context.fillStyle = 'rgba(0, 212, 255, 0.15)';
        context.fillText(text, size / 2, size / 2);

        // Main text
        context.shadowBlur = 10;
        context.fillStyle = 'rgba(0, 212, 255, 0.3)';
        context.fillText(text, size / 2, size / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;

        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        const sprite = new THREE.Sprite(material);
        sprite.scale.set(100, 100, 1);

        return sprite;
    }

    createGrid() {
        // Create a subtle grid on the floor
        const gridHelper = new THREE.GridHelper(2000, 40, 0x00d4ff, 0x0a0a0f);
        gridHelper.position.y = -400;
        gridHelper.material.opacity = 0.1;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        // Add some wireframe geometric shapes
        const geometries = [
            new THREE.IcosahedronGeometry(50, 0),
            new THREE.OctahedronGeometry(40, 0),
            new THREE.TetrahedronGeometry(35, 0)
        ];

        this.wireframes = [];

        geometries.forEach((geo, i) => {
            const edges = new THREE.EdgesGeometry(geo);
            const material = new THREE.LineBasicMaterial({
                color: 0x00d4ff,
                transparent: true,
                opacity: 0.2
            });
            const wireframe = new THREE.LineSegments(edges, material);

            wireframe.position.x = (i - 1) * 400;
            wireframe.position.y = Math.random() * 200 - 100;
            wireframe.position.z = -300;

            wireframe.userData = {
                rotationSpeed: {
                    x: (Math.random() - 0.5) * 0.01,
                    y: (Math.random() - 0.5) * 0.01
                }
            };

            this.wireframes.push(wireframe);
            this.scene.add(wireframe);
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const time = this.clock.getElapsedTime();

        // Rotate particles slowly
        if (this.particles) {
            this.particles.rotation.y = time * 0.02;
            this.particles.rotation.x = Math.sin(time * 0.1) * 0.1;
        }

        // Animate floating symbols
        this.symbols.forEach(symbol => {
            const { originalY, speed, phase } = symbol.userData;
            symbol.position.y = originalY + Math.sin(time * speed + phase) * 30;
            symbol.material.rotation += symbol.userData.rotationSpeed;
        });

        // Animate wireframes
        this.wireframes.forEach(wireframe => {
            wireframe.rotation.x += wireframe.userData.rotationSpeed.x;
            wireframe.rotation.y += wireframe.userData.rotationSpeed.y;
        });

        // Camera follows mouse slightly
        this.camera.position.x += (this.mouseX * 0.1 - this.camera.position.x) * 0.02;
        this.camera.position.y += (-this.mouseY * 0.1 + 100 - this.camera.position.y) * 0.02;
        this.camera.lookAt(this.scene.position);

        this.renderer.render(this.scene, this.camera);
    }

    bindEvents() {
        // Resize handler
        window.addEventListener('resize', () => {
            this.width = window.innerWidth;
            this.height = window.innerHeight;

            this.camera.aspect = this.width / this.height;
            this.camera.updateProjectionMatrix();

            this.renderer.setSize(this.width, this.height);
        });

        // Mouse movement
        document.addEventListener('mousemove', (event) => {
            this.mouseX = (event.clientX - this.width / 2);
            this.mouseY = (event.clientY - this.height / 2);
        });
    }
}

// Initialize scene when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new QuantScene();
});
