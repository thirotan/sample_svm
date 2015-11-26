# -*- coding: utf-8 -*-
require 'libsvm'
require 'gnuplot'

# 点クラス
class Point
  attr_reader :x, :y, :label
  def initialize(x, y)
    @x, @y = x, y
    @label = calc_label(x, y)
  end
  def calc_label(x, y)
    fx = x + 50*Math.sin(x/15.0)
    (y > fx) ? 1 : 2
  end
end

# 点の初期化
points = Array.new(5000){ Point.new(rand(300), rand(300)) }
# pointsを見てみる
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title("points")
    plot.size("ratio 1 1")
    plot.xlabel("x")
    plot.ylabel("y")
    plot.xrange("[0:300]")
    plot.yrange("[0:300]")

    p1, p2 = points.partition{|p| p.label == 1}

    plot.data << Gnuplot::DataSet.new([p1.collect(&:x), p1.collect(&:y)]) 
    plot.data << Gnuplot::DataSet.new([p2.collect(&:x), p2.collect(&:y)]) 

    plot.data << Gnuplot::DataSet.new("x + 50*sin(x/15)") do |d|
      d.with = "line"
      d.notitle
    end
  end
end


# SVMパラメータ
parameter = Libsvm::SvmParameter.new.tap { |p|
  p.svm_type = 0 # C_SVC
  p.kernel_type = 2 # RBF
  p.cache_size = 100 # in MB
  p.eps = 0.000001
  p.degree = 3
  p.c = 1
  p.nu = 0.5
  p.gamma = 0.001
  p.p = 0.1
}

# 教師データ（座標）
examples = points.collect {|p| 
  Libsvm::Node.features(p.x, p.y) 
}
# 教師データの応答（答え）
labels = points.collect(&:label)

# 訓練
problem = Libsvm::Problem.new
problem.set_examples(labels, examples)
model = Libsvm::Model.train(problem, parameter)
model.save("sample_train.txt")

Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title("points")
    plot.size("ratio 1 1")
    plot.xlabel("x")
    plot.ylabel("y")
    plot.xrange("[0:300]")
    plot.yrange("[0:300]")

    p1 = []
    p2 = []

    for x in 0...300
      for y in 0...300
        query = Libsvm::Node.features(x, y)
        pred = model.predict(query)
        if pred == 1
          p1 << Point.new(x, y)
        elsif pred == 2
          p2 << Point.new(x, y)
        end
      end
    end

    plot.data << Gnuplot::DataSet.new([p1.collect(&:x), p1.collect(&:y)]) do |d|
      d.with = "dot"
    end
    plot.data << Gnuplot::DataSet.new([p2.collect(&:x), p2.collect(&:y)]) do |d|
      d.with = "dot"
    end

  end
end
