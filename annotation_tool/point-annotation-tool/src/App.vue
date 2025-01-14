<template>
  <div class="container">
    <h1>Point Annotation Tool</h1>
    <div class="controls">
      <div class="page-info">
        <span>Current Page: {{ currentPage }}</span>
        <span class="label-info">Current Label: {{ currentLabel }}</span>
        <button @click="saveLabelsSwitchPage">Save & Next Page</button>
      </div>
      <div class="instructions">
        <p>Hold Ctrl and hover over points to annotate them. Release Ctrl to move to next label.</p>
      </div>
    </div>
    
    <div 
      class="canvas-container" 
      @mousemove="handleMouseMove"
      @keydown.x="startAnnotation"
      @keyup.x="finishAnnotation"
      @keydown.f="startFootnoteAnnotation"
      @keyup.f="finishFootnoteAnnotation"
      tabindex="0"
    >
      <canvas ref="canvas"></canvas>
    </div>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      currentPage: 0,
      points: [],
      labels: [],
      isCtrlPressed: false,
      ctx: null,
      currentLabel: 0,
      tempLabel: 0,
      margin: 50  // margin in pixels
    }
  },
  mounted() {
    this.ctx = this.$refs.canvas.getContext('2d')
    this.loadPoints()
  },
  methods: {
    async loadPoints() {
      try {
        const response = await fetch(`/test-data/pg_${this.currentPage}_points.txt`)
        const text = await response.text()
        this.points = text.trim().split('\n').map(line => {
          const [x, y] = line.split(' ').map(Number)
          return { x, y, label: null }
        })
        this.labels = new Array(this.points.length).fill(null)
        this.updateCanvasSize()
        this.drawPoints()
      } catch (error) {
        console.error('Error loading points:', error)
      }
    },
    
    updateCanvasSize() {
      const xCoords = this.points.map(p => p.x)
      const yCoords = this.points.map(p => p.y)
      const minX = Math.min(...xCoords)
      const maxX = Math.max(...xCoords)
      const minY = Math.min(...yCoords)
      const maxY = Math.max(...yCoords)
      
      // Calculate dimensions with margins
      let width = maxX - minX + (this.margin * 2)
      let height = maxY - minY + (this.margin * 2)
      
      // Get container dimensions
      const container = document.querySelector('.container')
      const maxWidth = container.clientWidth - 40  // subtract padding
      const maxHeight = window.innerHeight - 200   // subtract space for controls
      
      // Calculate scale if needed
      const scale = Math.min(
        maxWidth / width,
        maxHeight / height,
        1  // Don't scale up, only down
      )
      
      // Apply scale
      width *= scale
      height *= scale
      
      // Update canvas size
      const canvas = this.$refs.canvas
      canvas.width = width
      canvas.height = height
      
      // Adjust point coordinates
      const scaleX = (width - this.margin * 2) / (maxX - minX)
      const scaleY = (height - this.margin * 2) / (maxY - minY)
      this.points = this.points.map(point => ({
        ...point,
        x: (point.x - minX) * scaleX + this.margin,
        y: (point.y - minY) * scaleY + this.margin
      }))
    },
    
    drawPoints() {
      this.ctx.clearRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height)
      
      // Draw all points
      this.points.forEach((point, index) => {
        this.ctx.beginPath()
        this.ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
        
        if (this.labels[index] !== null) {
          const hue = (this.labels[index] * 137.5) % 360
          this.ctx.fillStyle = `hsl(${hue}, 70%, 50%)`
        } else {
          this.ctx.fillStyle = 'gray'
        }
        
        this.ctx.fill()
      })
    },
    
    handleMouseMove(event) {
      if (!this.isCtrlPressed) return
      
      const rect = this.$refs.canvas.getBoundingClientRect()
      const scale = rect.width / this.$refs.canvas.width
      const x = (event.clientX - rect.left) / scale
      const y = (event.clientY - rect.top) / scale
      
      // Find and label the closest point
      const closePoint = this.points.findIndex(point => {
        const distance = Math.sqrt(
          Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)
        )
        return distance < 10
      })
      
      if (closePoint !== -1) {
        this.labels[closePoint] = this.currentLabel
        this.drawPoints()
      }
    },
    
    startAnnotation() {
      this.isCtrlPressed = true
    },
    
    finishAnnotation() {
      this.isCtrlPressed = false
      this.currentLabel += 1  // Increment label for next annotation
    },

    startFootnoteAnnotation() {
      this.isCtrlPressed = true
      this.tempLabel = this.currentLabel
      this.currentLabel = 50
    },
    
    finishFootnoteAnnotation() {
      this.isCtrlPressed = false
      this.currentLabel = this.tempLabel  // Increment label for next annotation

    },
    
    async saveLabelsSwitchPage() {
      const labelsText = this.labels.join('\n')
      console.log(`Saving to pg_${this.currentPage}_labels.txt:`)
      console.log(labelsText)
      
      this.currentPage++
      this.currentLabel = 0  // Reset label counter for new page
      await this.loadPoints()
    }
  }
}
</script>

<style>
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

.controls {
  margin-bottom: 20px;
}

.page-info {
  margin-bottom: 10px;
  display: flex;
  gap: 20px;
  align-items: center;
}

.label-info {
  font-weight: bold;
  color: #333;
}

.instructions {
  color: #666;
}

.canvas-container {
  border: 1px solid #ccc;
  outline: none;
}

canvas {
  display: block;
}

button {
  padding: 8px 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #45a049;
}
</style>