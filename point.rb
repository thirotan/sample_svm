require 'libsvm'
require 'gnuplot'

class Point
  attr_reader :x, :y, :label

  def initialize(x,y)
    @x, @y = x,y
    @label = calc_label(x,y)
  end

  def calc_label(x,y)
    fx = x
    (y > fx) ? 1 : 2
  end
end


class SVM
  def set_params(params={})
    parameter = Libsvm::SvmParameter.new.tap {|p|
      p.svm_type = 0
      p.kernel_type = 2
      p.cache_size = 100
      p.eps = 0.000001
      p.degree = 3
      p.gamma = 0.001
      p.c = 1
      p.nu = 0.5
      p.p = 0.1
    }
    return parameter.merge(params)
  end 

  def train
    parameter = set_params
    points = Array.new(500){ Point.new(rand(300), rand(300))}
    examples = points.collect{ |p|
      Libsvm::Node.features(p.x,  p.y)
    }
    labels = points.collect(&:label)
    problem = Libsvm::Problem.new
    problem.set_examples(labels,examples)
    model = Libsvm::Model.train(problem,parameter)
    return model
  end
end

svm = SVM.new
#puts "result: #{svm.train}"

points = Array.new(500){ Point.new(rand(300), rand(300)) }

Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title("points")
    plot.size("ratio 1 1")
    plot.xlabel("x")
    plot.ylabel("y")
    plot.xrange("[0:300]")
    plot.yrange("[0:300]")

    p1, p2 = points.partition{|p| p.label == 1 }

    plot.data << Gnuplot::Dataset.new([p1.collect(&:x), p1.collect(&:y)])
    plot.data << Gnuplot::Dataset.new([p2.collect(&:x), p2.collect(&:y)])
    plot.data << Gnuplot::Dataset.new("x") do |d|
      d.with = "line"
      d.notitle
    end
  end
end
