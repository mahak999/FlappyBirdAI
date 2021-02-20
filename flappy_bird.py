import pygame
import neat
import time
import os
import random
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "background.png")))
STAT_FONT = pygame.font.SysFont("Comic Sans MS", 40)


class Bird:
	IMGS = BIRD_IMGS
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5

	def __init__(self, x, y):
		self.x = x # initial position of bird measured in horizontal direction
		self.y = y # initial position of bird measured in vertical direction
		self.tilt = 0 # initial tilt set to 0
		self.tick_count = 0 # tick count starts at 0
		self.vel = 0 # velocity of the bird set to 0 for the first frame
		self.height = self.y # distance of the bird from the ground
		self.img_count = 0 # count of the image being currently shown
		self.img = self.IMGS[0] # first bird image being shown

	def jump(self):
		self.vel = -10.5 # bird jumps upwards as the game starts i.e. negative velocity indicates moving upwards
		self.tick_count = 0 # setting tick count to 0 to keep tract pf the jump
		self.height = self.y

	def move(self):
		self.tick_count += 1 # number of times the bird has moved since the last jump
		d = self.vel*self.tick_count + 1.5*self.tick_count**2 # used to find the uipward movement using the equation of motion

		if d>= 16:
			d = 16
		if d<0:
			d -= 2
		self.y = self.y +d

		if d<0 or self.y < self.height + 50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		else:
			if self.tilt < -90:
				self.tilt -= self.ROT_VEL


	def draw(self, win):
		self.img_count += 1

		if self.img_count < self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.img_count < self.ANIMATION_TIME*2:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME*3:
			self.img = self.IMGS[2]
		elif self.img_count < self.ANIMATION_TIME*4:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME*4 + 1:
			self.img = self.IMGS[0]
			self.img_count = 0

		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image, new_rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img) # to compare it with other masks like that of a pipe



class Pipe:
	GAP = 200
	VEL = 5

	def __init__(self, x):
		self.x = x
		self.height = 0

		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
		self.PIPE_BOTTOM = PIPE_IMG

		self.passed = False
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x -= self.VEL # moving the pipe to the left so that it looks like frames are changing

	def draw(self,win):
		win.blit(self.PIPE_TOP, (self.x, self.top))
		win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

	def collide(self, bird):
		bird_mask = bird.get_mask() # getting the bird's mask
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		# calculating offset

		top_offset = (self.x - bird.x, self.top - round(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

		# colliding condititons
		b_point = bird_mask.overlap(bottom_mask, bottom_offset)
		t_point = bird_mask.overlap(top_mask, top_offset)

		if t_point or b_point: # if any of them is not null
			return True

		return False


class Base:
	VEL = 5
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	def move(self):
		self.x1 -= self.VEL
		self.x2 -= self.VEL

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH

		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH

	def draw(self, win):
		win.blit(self.IMG, (self.x1, self.y))
		win.blit(self.IMG, (self.x2, self.y))



def draw_window(win, birds, pipes, base, score, gen):
	win.blit(BG_IMG, (0,0))
	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (237, 255, 144))
	win.blit(text, (WIN_WIDTH - 10- text.get_width(), 10))

	text = STAT_FONT.render("Gen: " + str(gen), 1, (200, 255, 144))
	win.blit(text, (10,10))

	base.draw(win)
	for bird in birds:
		bird.draw(win)

	pygame.display.update()


def main(genomes, config):
	# bird = [Bird(230, 350)]
	global GEN
	GEN += 1
	birds = []
	ge = []
	nets = []

	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		birds.append(Bird(230, 350))
		g.fitness = 0
		ge.append(g)

	base = Base(730) # bottom of the screen is 800
	pipes = [Pipe(600)]
	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()
	score = 0
	run = True

	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				
		pipe_ind = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind = 1
		else:
			run = False
			break
		
		for x, bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1

			output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

			if output[0] > 0.5:
				bird.jump()

		add_pipe = False
		rem_list = []

		for pipe in pipes:
			pipe.move()
			for x, bird in enumerate(birds):
				if pipe.collide(bird):
					ge[x].fitness -= 1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed  = True
					add_pipe = True
			
			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem_list.append(pipe)
			

		if add_pipe:
			score +=  1
			for g in ge:
				g.fitness += 5
			pipes.append(Pipe(600))

		for each_pipe in rem_list:
			pipes.remove(each_pipe)
		
		for x, bird in enumerate(birds):
			if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)


		base.move()
		draw_window(win, birds, pipes, base, score, GEN)



def run(config_file):

	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
	 	neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(main, 50) # fitness function set

if __name__ == "__main__":
	local_directory = os.path.dirname(__file__) # current directory loaded
	config_path = os.path.join(local_directory, "NEAT_config.txt")
	run(config_path)

